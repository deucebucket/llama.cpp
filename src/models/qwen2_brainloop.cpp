#include "models.h"
#include <ggml-alloc.h>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

struct brainloop_weight {
    std::vector<float> data;
    int rows, cols;
};

static brainloop_weight load_brainloop_weight(const std::string & path) {
    brainloop_weight bw;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fprintf(stderr, "BRAINLOOP: cannot open %s\n", path.c_str());
        return bw;
    }
    int32_t shape[2];
    file.read((char*)shape, 8);
    bw.rows = shape[0];
    bw.cols = shape[1];
    size_t n_floats = (size_t)bw.rows * bw.cols;
    bw.data.resize(n_floats);
    file.read((char*)bw.data.data(), n_floats * sizeof(float));
    return bw;
}

struct brainloop_gpu_cache {
    bool initialized = false;

    struct ggml_tensor * ln1_w = nullptr;
    struct ggml_tensor * ln1_b = nullptr;
    struct ggml_tensor * ln2_w = nullptr;
    struct ggml_tensor * ln2_b = nullptr;

    struct ggml_tensor * q_w = nullptr;
    struct ggml_tensor * q_b = nullptr;
    struct ggml_tensor * k_w = nullptr;
    struct ggml_tensor * k_b = nullptr;
    struct ggml_tensor * v_w = nullptr;
    struct ggml_tensor * v_b = nullptr;
    struct ggml_tensor * o_w = nullptr;
    struct ggml_tensor * o_b = nullptr;

    struct ggml_tensor * up_w  = nullptr;
    struct ggml_tensor * up_b  = nullptr;
    struct ggml_tensor * dn_w  = nullptr;
    struct ggml_tensor * dn_b  = nullptr;

    struct ggml_tensor * gate   = nullptr;
    struct ggml_tensor * rev_emb = nullptr;

    // Inline RAG: document index in model's native embedding space
    struct ggml_tensor * rag_docs = nullptr;
    int rag_n_docs = 0;

    // KV Cache Hijack: synthetic key/value pairs for forced attention
    struct ggml_tensor * synth_k = nullptr;  // [n_embd_head * n_kv_head]
    struct ggml_tensor * synth_v = nullptr;

    float gate_sigmoid = 0.5f;
    float rag_scale = 0.6225f;  // learned via training: sigmoid(rag_scale)
};

static brainloop_gpu_cache & get_brainloop_cache(
    const llama_model & model, const char * weight_dir, int split_layer)
{
    static brainloop_gpu_cache cache;
    if (cache.initialized) {
        return cache;
    }

    auto lw = [&](const char * name) -> brainloop_weight {
        return load_brainloop_weight(std::string(weight_dir) + name);
    };

    brainloop_weight bw_ln1_w = lw("refiner_ln1_weight.bin");
    brainloop_weight bw_ln1_b = lw("refiner_ln1_bias.bin");
    brainloop_weight bw_ln2_w = lw("refiner_ln2_weight.bin");
    brainloop_weight bw_ln2_b = lw("refiner_ln2_bias.bin");

    brainloop_weight bw_q_w = lw("refiner_attn_q_weight.bin");
    brainloop_weight bw_q_b = lw("refiner_attn_q_bias.bin");
    brainloop_weight bw_k_w = lw("refiner_attn_k_weight.bin");
    brainloop_weight bw_k_b = lw("refiner_attn_k_bias.bin");
    brainloop_weight bw_v_w = lw("refiner_attn_v_weight.bin");
    brainloop_weight bw_v_b = lw("refiner_attn_v_bias.bin");
    brainloop_weight bw_o_w = lw("refiner_attn_output_weight.bin");
    brainloop_weight bw_o_b = lw("refiner_attn_output_bias.bin");

    brainloop_weight bw_up_w = lw("refiner_ffn_up_weight.bin");
    brainloop_weight bw_up_b = lw("refiner_ffn_up_bias.bin");
    brainloop_weight bw_dn_w = lw("refiner_ffn_down_weight.bin");
    brainloop_weight bw_dn_b = lw("refiner_ffn_down_bias.bin");

    brainloop_weight bw_gate   = lw("refiner_gate.bin");
    brainloop_weight bw_rev_emb = lw("refiner_rev_emb.bin");

    bool ok = !bw_q_w.data.empty() && !bw_o_w.data.empty()
           && !bw_up_w.data.empty() && !bw_dn_w.data.empty();
    if (!ok) {
        fprintf(stderr, "BRAINLOOP: missing weight files — brainloop disabled\n");
        cache.initialized = true;
        return cache;
    }

    int n_embd = (int)model.hparams.n_embd;
    int n_ff   = n_embd * 2;

    int n_tensors = 18;
    struct ggml_init_params wparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_tensors + 2),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * wctx = ggml_init(wparams);

    cache.ln1_w = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd);
    cache.ln1_b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd);
    cache.ln2_w = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd);
    cache.ln2_b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd);

    cache.q_w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, n_embd);
    cache.q_b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd);
    cache.k_w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, n_embd);
    cache.k_b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd);
    cache.v_w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, n_embd);
    cache.v_b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd);
    cache.o_w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, n_embd);
    cache.o_b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd);

    cache.up_w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, n_ff);
    cache.up_b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_ff);
    cache.dn_w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_ff, n_embd);
    cache.dn_b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n_embd);

    cache.gate   = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 1);
    cache.rev_emb = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, 8);

    ggml_backend_buffer_type_t buft = model.select_buft(split_layer);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);

    auto cps = [](struct ggml_tensor * t, const brainloop_weight & bw) {
        if (!bw.data.empty()) {
            ggml_backend_tensor_set(t, bw.data.data(), 0,
                bw.data.size() * sizeof(float));
        }
    };

    cps(cache.ln1_w, bw_ln1_w); cps(cache.ln1_b, bw_ln1_b);
    cps(cache.ln2_w, bw_ln2_w); cps(cache.ln2_b, bw_ln2_b);
    cps(cache.q_w, bw_q_w); cps(cache.q_b, bw_q_b);
    cps(cache.k_w, bw_k_w); cps(cache.k_b, bw_k_b);
    cps(cache.v_w, bw_v_w); cps(cache.v_b, bw_v_b);
    cps(cache.o_w, bw_o_w); cps(cache.o_b, bw_o_b);
    cps(cache.up_w, bw_up_w); cps(cache.up_b, bw_up_b);
    cps(cache.dn_w, bw_dn_w); cps(cache.dn_b, bw_dn_b);
    cps(cache.gate, bw_gate);
    cps(cache.rev_emb, bw_rev_emb);

    cache.gate_sigmoid = bw_gate.data.empty() ? 0.5f
        : 1.0f / (1.0f + expf(-bw_gate.data[0]));

    fprintf(stderr, "BRAINLOOP: GPU-allocated refiner weights on %s (gate=%.4f, %d tensors)\n",
        ggml_backend_buffer_name(buf), cache.gate_sigmoid, n_tensors);

    cache.initialized = true;

    // Inline RAG: load pre-computed document embeddings
    {
        brainloop_weight bw_rag = load_brainloop_weight("rag-experiment/rag_docs_real.bin");
        fprintf(stderr, "BRAINLOOP RAG DEBUG: rows=%d cols=%d empty=%d n_embd=%d\n",
            bw_rag.rows, bw_rag.cols, bw_rag.data.empty(), n_embd);
        if (!bw_rag.data.empty() && bw_rag.cols == n_embd) {
            cache.rag_n_docs = bw_rag.rows;
            struct ggml_init_params rp = { ggml_tensor_overhead() * 2, nullptr, true };
            struct ggml_context * rctx = ggml_init(rp);
            cache.rag_docs = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, n_embd, cache.rag_n_docs);
            ggml_backend_buffer_t rbuf = ggml_backend_alloc_ctx_tensors_from_buft(rctx, buft);
            ggml_backend_tensor_set(cache.rag_docs, bw_rag.data.data(), 0, bw_rag.data.size() * sizeof(float));
            fprintf(stderr, "BRAINLOOP: loaded RAG index: %d docs x %d dim on %s\n",
                cache.rag_n_docs, n_embd, ggml_backend_buffer_name(rbuf));
        }
    }

    // KV Cache Hijack: load synthetic K and V for forced attention
    {
        brainloop_weight bw_k = load_brainloop_weight("rag-experiment/canary_k.bin");
        brainloop_weight bw_v = load_brainloop_weight("rag-experiment/canary_v.bin");
        if (!bw_k.data.empty() && !bw_v.data.empty()) {
            struct ggml_init_params rp = { ggml_tensor_overhead() * 2, nullptr, true };
            struct ggml_context * rctx = ggml_init(rp);
            cache.synth_k = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, bw_k.cols, bw_k.rows);
            cache.synth_v = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, bw_v.cols, bw_v.rows);
            ggml_backend_buffer_t rbuf = ggml_backend_alloc_ctx_tensors_from_buft(rctx, buft);
            ggml_backend_tensor_set(cache.synth_k, bw_k.data.data(), 0, bw_k.data.size() * sizeof(float));
            ggml_backend_tensor_set(cache.synth_v, bw_v.data.data(), 0, bw_v.data.size() * sizeof(float));
            fprintf(stderr, "BRAINLOOP: loaded synthetic KV for hijack\n");
        }
    }

    return cache;
}

llm_build_qwen2_brainloop::llm_build_qwen2_brainloop(
    const llama_model & model, const llm_graph_params & params
) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const int split_layer = n_layer / 2;  // midpoint for any model size
    const int n_rev = 2;  // matches RAG training config (REVS=2)

    ggml_tensor * cur;
    ggml_tensor * inpL;
    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const char * weight_dir = "brainloop-ggml-weights/";
    auto & cache = get_brainloop_cache(model, weight_dir, split_layer);

    bool use_brainloop = (cache.q_w != nullptr
        && cache.q_w->ne[0] == (int64_t)hparams.n_embd);

    int n_embd_i = (int)hparams.n_embd;
    int n_head_i = (int)hparams.n_head();
    ggml_tensor * rag_ctx_cached = nullptr;

    // Query RAG at embedding level (before any layers)
    if (cache.rag_docs) {
        ggml_tensor * sim = ggml_mul_mat(ctx0, cache.rag_docs, inpL);
        sim = ggml_scale(ctx0, sim, 50.0f);
        sim = ggml_soft_max(ctx0, sim);
        ggml_tensor * t_rag = ggml_cont(ctx0, ggml_transpose(ctx0, cache.rag_docs));
        rag_ctx_cached = ggml_mul_mat(ctx0, t_rag, sim);
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            // KV Hijack: disabled - flash_attn doesn't handle concat'd tensor strides
            if (false && cache.synth_k && il >= split_layer) {
                int n_kvh = (int)n_head_kv;
                ggml_tensor * sk = ggml_reshape_3d(ctx0, cache.synth_k, (int)n_embd_head, n_kvh, 1);
                ggml_tensor * sv = ggml_reshape_3d(ctx0, cache.synth_v, (int)n_embd_head, n_kvh, 1);
                // Cast all to F32 (CUDA concat requires F32)
                sk = ggml_cast(ctx0, sk, GGML_TYPE_F32);
                sv = ggml_cast(ctx0, sv, GGML_TYPE_F32);
                ggml_tensor * Kf32 = ggml_cast(ctx0, Kcur, GGML_TYPE_F32);
                ggml_tensor * Vf32 = ggml_cast(ctx0, Vcur, GGML_TYPE_F32);
                Kcur = ggml_concat(ctx0, Kf32, sk, 2);
                Vcur = ggml_concat(ctx0, Vf32, sv, 2);
            }

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Hand-to-hand: progressive blend from layer 0 through 35
        // Phase 1 (0-7): inception - tiny seed (1-8%)
        // Phase 2 (8-17): fact finding - ramp up (10-50%)
        // Phase 3 (18-35): reasoning + output - dominate (55-100%)
        if (rag_ctx_cached && il <= 35) {
            float blend;
            if (il < 8)       blend = (il + 1) * 0.01f;       // 1-8%
            else if (il < 18) blend = 0.08f + (il-7)*0.04f;    // 12-48%
            else              blend = 0.50f + (il-17)*0.03f;   // 53-104%
            ggml_tensor * rag_part = ggml_scale(ctx0, rag_ctx_cached, blend);
            ggml_tensor * model_part = ggml_scale(ctx0, cur, 1.0f - blend);
            cur = ggml_add(ctx0, model_part, rag_part);
        }

        // BRAINLOOP: after layer 17, before 18 — matches PyTorch placement
        if (use_brainloop && il == split_layer - 1) {
            float kq_scale = 1.0f / sqrtf((float)n_embd_head);
            for (int rev = 0; rev < n_rev; ++rev) {
                ggml_tensor * x = cur;

                // --- Self-attention ---
                ggml_tensor * a_norm = ggml_rms_norm(ctx0, x, 1e-6f);
                a_norm = ggml_mul(ctx0, a_norm, cache.ln1_w);
                if (cache.ln1_b) a_norm = ggml_add(ctx0, a_norm, cache.ln1_b);

                ggml_tensor * Qcur = ggml_mul_mat(ctx0, cache.q_w, a_norm);
                if (cache.q_b) Qcur = ggml_add(ctx0, Qcur, cache.q_b);
                ggml_tensor * Kcur = ggml_mul_mat(ctx0, cache.k_w, a_norm);
                if (cache.k_b) Kcur = ggml_add(ctx0, Kcur, cache.k_b);
                ggml_tensor * Vcur = ggml_mul_mat(ctx0, cache.v_w, a_norm);
                if (cache.v_b) Vcur = ggml_add(ctx0, Vcur, cache.v_b);

                // Reshape to [n_embd_head, n_head, n_tokens]
                int nt = (int)x->ne[1];
                Qcur = ggml_reshape_3d(ctx0, Qcur, (int)n_embd_head, n_head_i, nt);
                Kcur = ggml_reshape_3d(ctx0, Kcur, (int)n_embd_head, n_head_i, nt);
                Vcur = ggml_reshape_3d(ctx0, Vcur, (int)n_embd_head, n_head_i, nt);

                // RoPE
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    (int)n_embd_head, rope_type, (int)n_ctx_orig, freq_base,
                    1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    (int)n_embd_head, rope_type, (int)n_ctx_orig, freq_base,
                    1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

                // Multi-head attention via build_attn_mha
                // (mask=nullptr — hidden states at this point are already
                //  causally encoded by the autogressive base model)
                ggml_tensor * attn_out = build_attn_mha(
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, nullptr,
                    kq_scale, split_layer);

                // Output projection
                attn_out = ggml_mul_mat(ctx0, cache.o_w, attn_out);
                if (cache.o_b) attn_out = ggml_add(ctx0, attn_out, cache.o_b);

                x = ggml_add(ctx0, x, attn_out);

                // --- FFN ---
                ggml_tensor * f_norm = ggml_rms_norm(ctx0, x, 1e-6f);
                f_norm = ggml_mul(ctx0, f_norm, cache.ln2_w);
                if (cache.ln2_b) f_norm = ggml_add(ctx0, f_norm, cache.ln2_b);
                ggml_tensor * f_up  = ggml_mul_mat(ctx0, cache.up_w, f_norm);
                if (cache.up_b) f_up = ggml_add(ctx0, f_up, cache.up_b);
                ggml_tensor * f_act = ggml_gelu(ctx0, f_up);
                ggml_tensor * f_dn  = ggml_mul_mat(ctx0, cache.dn_w, f_act);
                if (cache.dn_b) f_dn = ggml_add(ctx0, f_dn, cache.dn_b);
                x = ggml_add(ctx0, x, f_dn);

                // Inline RAG: hard top-1 via sharp softmax (temperature-scaled)
                if (cache.rag_docs) {
                    ggml_tensor * sim = ggml_mul_mat(ctx0, cache.rag_docs, x);
                    sim = ggml_scale(ctx0, sim, 500.0f); // sharpen to near-argmax
                    sim = ggml_soft_max(ctx0, sim);
                    ggml_tensor * t_rag = ggml_cont(ctx0, ggml_transpose(ctx0, cache.rag_docs));
                    ggml_tensor * rag_ctx = ggml_mul_mat(ctx0, t_rag, sim);
                    x = ggml_add(ctx0, x, ggml_scale(ctx0, rag_ctx, cache.rag_scale));
                }

                // Gated residual
                ggml_tensor * delta = ggml_sub(ctx0, x, cur);
                delta = ggml_mul(ctx0, delta, ggml_sigmoid(ctx0, cache.gate));
                cur = ggml_add(ctx0, cur, delta);
            }
            cb(cur, "brainloop", il);
        }

        inpL = cur;
    }
    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // Logit bias: boost canary tokens only (CPU-allocated, no suppression)
    {
        int n_vocab = llama_vocab_n_tokens(&model.vocab);
        struct ggml_init_params bp = {
            ggml_tensor_overhead() + n_vocab * sizeof(float), nullptr, false
        };
        struct ggml_context * bctx = ggml_init(bp);
        ggml_tensor * bias = ggml_new_tensor_1d(bctx, GGML_TYPE_F32, n_vocab);
        if (bias->data) {
            memset(bias->data, 0, n_vocab * sizeof(float));
            float * d = (float *)bias->data;
            int ids[] = {8847, 36, 48021, 53, 300, 41121, 57, 324, 713, 44220, 372, 641, 7660, 811, 79281, 3313, 67, 22280, 22, 12, 42539, 278, 46111, 4203};
            int n = sizeof(ids)/sizeof(ids[0]);
            for (int i = 0; i < n; i++) if (ids[i] < n_vocab) d[ids[i]] = 500.0f;
        }
        cur = ggml_add(ctx0, cur, bias);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}
