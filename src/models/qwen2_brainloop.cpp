#include "models.h"
#include <ggml-alloc.h>
#include "../llama-impl.h"
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

    // Cartridge: per-layer K/V from real forward pass
    struct ggml_tensor * cart_k = nullptr;  // [n_layers, n_kv_dim]
    struct ggml_tensor * cart_v = nullptr;

    float gate_sigmoid = 0.5f;
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

    // Load cartridge K/V from real model forward pass
    {
        brainloop_weight bw_k = load_brainloop_weight("rag-experiment/cartridge_k.bin");
        brainloop_weight bw_v = load_brainloop_weight("rag-experiment/cartridge_v.bin");
        if (!bw_k.data.empty() && !bw_v.data.empty()) {
            struct ggml_init_params rp = { ggml_tensor_overhead() * 2, nullptr, true };
            struct ggml_context * rctx = ggml_init(rp);
            cache.cart_k = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, bw_k.cols, bw_k.rows);
            cache.cart_v = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, bw_v.cols, bw_v.rows);
            ggml_backend_buffer_t rbuf = ggml_backend_alloc_ctx_tensors_from_buft(rctx, buft);
            ggml_backend_tensor_set(cache.cart_k, bw_k.data.data(), 0, bw_k.data.size() * sizeof(float));
            ggml_backend_tensor_set(cache.cart_v, bw_v.data.data(), 0, bw_v.data.size() * sizeof(float));
            fprintf(stderr, "BRAINLOOP: loaded cartridge K/V (%d layers x %d dim)\n", bw_k.rows, bw_k.cols);
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

    const int split_layer = 18;
    const int n_rev = 1;

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

            if (false && cache.cart_k && il >= split_layer) {
                struct ggml_init_params rp = { ggml_tensor_overhead() + sizeof(int32_t), nullptr, false };
                struct ggml_context * rctx = ggml_init(rp);
                ggml_tensor * row_idx = ggml_new_tensor_1d(rctx, GGML_TYPE_I32, 1);
                if (row_idx->data) ((int32_t*)row_idx->data)[0] = il;
                ggml_tensor * ck = ggml_get_rows(ctx0, cache.cart_k, row_idx);
                ggml_tensor * cv = ggml_get_rows(ctx0, cache.cart_v, row_idx);
                int n_kvh = (int)n_head_kv;
                ck = ggml_reshape_3d(ctx0, ck, (int)n_embd_head, n_kvh, 1);
                cv = ggml_reshape_3d(ctx0, cv, (int)n_embd_head, n_kvh, 1);
                ck = ggml_cast(ctx0, ck, GGML_TYPE_F16);
                cv = ggml_cast(ctx0, cv, GGML_TYPE_F16);
                // Manually expand from n_kvh=2 to n_head=16 via concat replication
                // Concat 8 copies of the 2-head tensor: 2*8 = 16 heads
                ggml_tensor * ck16 = ck; ggml_tensor * cv16 = cv;
                int reps = (int)n_head / n_kvh;  // 16/2 = 8
                for (int r = 1; r < reps; r++) {
                    ck16 = ggml_concat(ctx0, ck16, ck, 1);  // expand along head dim
                    cv16 = ggml_concat(ctx0, cv16, cv, 1);
                }
                // Manual attention: cartridge has 1 token, Q@K^T is just dot product
                // Qp: [128, n_tokens, 16], ck: [128, 1, 16], cv: [128, 1, 16]
                // scores = Q @ K^T: reduce over dim 0 -> [n_tokens, 16, 1]
                // We can use: scores = Q * K (element-wise) -> sum over dim 0 -> softmax over n_tokens
                // Skip complex attention, just add cartridge info directly to hidden state
                // The cartridge's output projection already carries semantic information
                cart_out = ggml_mul_mat(ctx0, cart_out, ggml_reshape_2d(ctx0, Qcur, n_embd_i, (int)Qcur->ne[1]*(int)Qcur->ne[2]));
                ggml_tensor * cart_proj = build_lora_mm(model.layers[il].wo, cart_out);
                if (model.layers[il].wo_b) cart_proj = ggml_add(ctx0, cart_proj, model.layers[il].wo_b);
                cart_proj = ggml_cont(ctx0, ggml_transpose(ctx0, cart_proj));
                cur = ggml_add(ctx0, cur, ggml_scale(ctx0, cart_proj, 0.001f));
            }
            // Normal attention
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), il);

            // Cartridge attention: compute Q's attention over cartridge K/V only
            if (cache.cart_k && il >= split_layer) {
                fprintf(stderr, "CART: layer %d cartridge start\n", il);
                struct ggml_init_params rp = { ggml_tensor_overhead() + sizeof(int32_t), nullptr, false };
                struct ggml_context * rctx = ggml_init(rp);
                ggml_tensor * row_idx = ggml_new_tensor_1d(rctx, GGML_TYPE_I32, 1);
                if (row_idx->data) ((int32_t*)row_idx->data)[0] = il;
                ggml_tensor * ck = ggml_get_rows(ctx0, cache.cart_k, row_idx);
                ggml_tensor * cv = ggml_get_rows(ctx0, cache.cart_v, row_idx);
                int n_kvh = (int)n_head_kv;
                ck = ggml_reshape_3d(ctx0, ck, (int)n_embd_head, n_kvh, 1);
                cv = ggml_reshape_3d(ctx0, cv, (int)n_embd_head, n_kvh, 1);
                ck = ggml_cast(ctx0, ck, GGML_TYPE_F16);
                cv = ggml_cast(ctx0, cv, GGML_TYPE_F16);
                // Use same Q, but only attend to cartridge K/V
                ggml_tensor * Qp = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                ck = ggml_permute(ctx0, ck, 0, 2, 1, 3);
                cv = ggml_permute(ctx0, cv, 0, 2, 1, 3);
                ggml_tensor * cart_out = ggml_flash_attn_ext(ctx0, Qp, ck, cv, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), 0.0f, 0.0f);
                cb(cart_out, LLAMA_TENSOR_NAME_FATTN, il);
                cart_out = ggml_reshape_2d(ctx0, cart_out, cart_out->ne[0]*cart_out->ne[1], cart_out->ne[2]*cart_out->ne[3]);
                // Project cartridge attention output and add to main output
                ggml_tensor * cart_proj = build_lora_mm(model.layers[il].wo, cart_out);
                if (model.layers[il].wo_b) cart_proj = ggml_add(ctx0, cart_proj, model.layers[il].wo_b);
                // ggml_mul_mat produces transposed output, transpose back for add
                cart_proj = ggml_cont(ctx0, ggml_transpose(ctx0, cart_proj));
                cur = ggml_add(ctx0, cur, ggml_scale(ctx0, cart_proj, 0.001f));
            }
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

    ggml_build_forward_expand(gf, cur);
}
