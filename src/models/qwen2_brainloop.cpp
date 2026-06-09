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
    return cache;
}

static ggml_tensor * brainloop_refine_pass(
    ggml_context * ctx0,
    ggml_tensor * hidden,
    int rev,
    ggml_tensor * ln1_w, ggml_tensor * ln1_b,
    ggml_tensor * ln2_w, ggml_tensor * ln2_b,
    ggml_tensor * attn_q_w, ggml_tensor * attn_q_b,
    ggml_tensor * attn_k_w, ggml_tensor * attn_k_b,
    ggml_tensor * attn_v_w, ggml_tensor * attn_v_b,
    ggml_tensor * attn_o_w, ggml_tensor * attn_o_b,
    ggml_tensor * ffn_up_w, ggml_tensor * ffn_up_b,
    ggml_tensor * ffn_down_w, ggml_tensor * ffn_down_b,
    ggml_tensor * gate,
    ggml_tensor * rev_emb,
    int n_embd, int n_head, int n_embd_head,
    ggml_tensor * inp_pos,
    float freq_base, int n_ctx_orig, int rope_type
) {
    int n_tokens = (int)hidden->ne[1];

    ggml_tensor * x = hidden;

    // Revolution embedding
    {
        ggml_tensor * rev_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        if (rev_idx->data) {
            ((int32_t *)rev_idx->data)[0] = rev;
        }
        ggml_tensor * rev_vec = ggml_get_rows(ctx0, rev_emb, rev_idx);
        x = ggml_add(ctx0, x, ggml_repeat(ctx0, rev_vec, x));
    }

    // Self-Attention with RMS norm + residual
    ggml_tensor * normed = ggml_rms_norm(ctx0, x, 1e-6f);
    normed = ggml_mul(ctx0, normed, ln1_w);
    if (ln1_b) normed = ggml_add(ctx0, normed, ln1_b);

    ggml_tensor * Qcur = ggml_mul_mat(ctx0, attn_q_w, normed);
    if (attn_q_b) Qcur = ggml_add(ctx0, Qcur, attn_q_b);
    ggml_tensor * Kcur = ggml_mul_mat(ctx0, attn_k_w, normed);
    if (attn_k_b) Kcur = ggml_add(ctx0, Kcur, attn_k_b);
    ggml_tensor * Vcur = ggml_mul_mat(ctx0, attn_v_w, normed);
    if (attn_v_b) Vcur = ggml_add(ctx0, Vcur, attn_v_b);

    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head, n_tokens);

    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
        n_embd_head, rope_type, n_ctx_orig, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
        n_embd_head, rope_type, n_ctx_orig, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * KQV = ggml_flash_attn_ext(ctx0, Qcur, Kcur, Vcur, nullptr,
        1.0f / sqrtf((float)n_embd_head), 0.0f, 0.0f);

    KQV = ggml_reshape_2d(ctx0, KQV, n_embd, n_tokens);

    ggml_tensor * attn_out = ggml_mul_mat(ctx0, attn_o_w, KQV);
    if (attn_o_b) attn_out = ggml_add(ctx0, attn_out, attn_o_b);

    x = ggml_add(ctx0, x, attn_out);

    // FFN with RMS norm + residual
    ggml_tensor * normed2 = ggml_rms_norm(ctx0, x, 1e-6f);
    normed2 = ggml_mul(ctx0, normed2, ln2_w);
    if (ln2_b) normed2 = ggml_add(ctx0, normed2, ln2_b);

    ggml_tensor * ffn_hidden = ggml_mul_mat(ctx0, ffn_up_w, normed2);
    if (ffn_up_b) ffn_hidden = ggml_add(ctx0, ffn_hidden, ffn_up_b);
    ffn_hidden = ggml_gelu(ctx0, ffn_hidden);
    ggml_tensor * ffn_out = ggml_mul_mat(ctx0, ffn_down_w, ffn_hidden);
    if (ffn_down_b) ffn_out = ggml_add(ctx0, ffn_out, ffn_down_b);

    x = ggml_add(ctx0, x, ffn_out);

    // Gated residual
    ggml_tensor * delta = ggml_sub(ctx0, x, hidden);
    ggml_tensor * gate_val = ggml_sigmoid(ctx0, gate);
    delta = ggml_mul(ctx0, gate_val, delta);
    x = ggml_add(ctx0, hidden, delta);

    return x;
}

llm_build_qwen2_brainloop::llm_build_qwen2_brainloop(
    const llama_model & model, const llm_graph_params & params
) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const int split_layer = 18;
    const int n_rev = 2;

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

        // BRAINLOOP: full attention+FFN refiner at split layer
        if (use_brainloop && il == split_layer) {
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

                // Multi-head attention via build_attn_mha (no explicit mask;
                // flash attention with 3D tensors is bidirectional by default)
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
