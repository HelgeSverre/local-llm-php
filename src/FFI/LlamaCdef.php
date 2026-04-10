<?php

declare(strict_types=1);

namespace HelgeSverre\LocalLlm\FFI;

final class LlamaCdef
{
    public static function definitions(): string
    {
        return <<<'CDEF'
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef unsigned long size_t;
typedef _Bool bool;

typedef void * ggml_backend_dev_t;
typedef void * ggml_backend_buffer_type_t;
typedef void * ggml_threadpool_t;
typedef void * llama_memory_t;

typedef int ggml_numa_strategy;
typedef int ggml_type;
typedef int ggml_log_level;
typedef int llama_flash_attn_type;

struct ggml_tensor;
struct gguf_context;
struct llama_vocab;
struct llama_model;
struct llama_context;
struct llama_sampler;
struct mtmd_context;
struct mtmd_bitmap;
struct mtmd_input_chunks;
struct llama_model_kv_override;
struct llama_model_tensor_buft_override;
struct llama_sampler_seq_config;

typedef int32_t llama_pos;
typedef int32_t llama_token;
typedef int32_t llama_seq_id;
typedef int mtmd_input_chunk_type;

typedef bool (*llama_progress_callback)(float progress, void * user_data);
typedef bool (*ggml_abort_callback)(void * user_data);
typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * tensor, bool ask, void * user_data);
typedef void (*ggml_log_callback)(ggml_log_level level, const char * text, void * user_data);

struct llama_model_params {
    ggml_backend_dev_t * devices;
    const struct llama_model_tensor_buft_override * tensor_buft_overrides;
    int32_t n_gpu_layers;
    int split_mode;
    int32_t main_gpu;
    const float * tensor_split;
    llama_progress_callback progress_callback;
    void * progress_callback_user_data;
    const struct llama_model_kv_override * kv_overrides;
    bool vocab_only;
    bool use_mmap;
    bool use_direct_io;
    bool use_mlock;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
};

struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t n_threads;
    int32_t n_threads_batch;
    int rope_scaling_type;
    int pooling_type;
    int attention_type;
    int flash_attn_type;
    float rope_freq_base;
    float rope_freq_scale;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float defrag_thold;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    ggml_type type_k;
    ggml_type type_v;
    ggml_abort_callback abort_callback;
    void * abort_callback_data;
    bool embeddings;
    bool offload_kqv;
    bool no_perf;
    bool op_offload;
    bool swa_full;
    bool kv_unified;
    struct llama_sampler_seq_config * samplers;
    size_t n_samplers;
};

typedef struct llama_sampler_chain_params {
    bool no_perf;
} llama_sampler_chain_params;

typedef struct llama_chat_message {
    const char * role;
    const char * content;
} llama_chat_message;

typedef struct mtmd_input_text {
    const char * text;
    bool add_special;
    bool parse_special;
} mtmd_input_text;

typedef struct llama_batch {
    int32_t n_tokens;
    llama_token * token;
    float * embd;
    llama_pos * pos;
    int32_t * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t * logits;
} llama_batch;

struct llama_perf_context_data {
    double t_start_ms;
    double t_load_ms;
    double t_p_eval_ms;
    double t_eval_ms;
    int32_t n_p_eval;
    int32_t n_eval;
    int32_t n_reused;
};

struct llama_perf_sampler_data {
    double t_sample_ms;
    int32_t n_sample;
};

struct mtmd_context_params {
    bool use_gpu;
    bool print_timings;
    int n_threads;
    const char * image_marker;
    const char * media_marker;
    int flash_attn_type;
    bool warmup;
    int image_min_tokens;
    int image_max_tokens;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};

void ggml_backend_load_all(void);

struct llama_model_params llama_model_default_params(void);
struct llama_context_params llama_context_default_params(void);
llama_sampler_chain_params llama_sampler_chain_default_params(void);
void llama_backend_init(void);
void llama_backend_free(void);
void llama_numa_init(ggml_numa_strategy numa);
int64_t llama_time_us(void);
void llama_log_set(ggml_log_callback log_callback, void * user_data);
const char * mtmd_default_marker(void);
struct mtmd_context_params mtmd_context_params_default(void);
struct mtmd_context * mtmd_init_from_file(const char * mmproj_fname, const struct llama_model * text_model, const struct mtmd_context_params ctx_params);
void mtmd_free(struct mtmd_context * ctx);
bool mtmd_support_vision(struct mtmd_context * ctx);
bool mtmd_support_audio(struct mtmd_context * ctx);
int mtmd_get_audio_sample_rate(struct mtmd_context * ctx);
struct mtmd_bitmap * mtmd_helper_bitmap_init_from_file(struct mtmd_context * ctx, const char * fname);
void mtmd_bitmap_free(struct mtmd_bitmap * bitmap);
void mtmd_bitmap_set_id(struct mtmd_bitmap * bitmap, const char * id);
struct mtmd_input_chunks * mtmd_input_chunks_init(void);
void mtmd_input_chunks_free(struct mtmd_input_chunks * chunks);
int32_t mtmd_tokenize(
    struct mtmd_context * ctx,
    struct mtmd_input_chunks * output,
    const mtmd_input_text * text,
    const struct mtmd_bitmap ** bitmaps,
    size_t n_bitmaps
);
int32_t mtmd_helper_eval_chunks(
    struct mtmd_context * ctx,
    struct llama_context * lctx,
    const struct mtmd_input_chunks * chunks,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * new_n_past
);
size_t mtmd_helper_get_n_tokens(const struct mtmd_input_chunks * chunks);

struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params);
void llama_model_free(struct llama_model * model);
struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params);
void llama_free(struct llama_context * ctx);

const struct llama_model * llama_get_model(const struct llama_context * ctx);
llama_memory_t llama_get_memory(const struct llama_context * ctx);
void llama_memory_clear(llama_memory_t mem, bool data);

const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
int32_t llama_model_n_ctx_train(const struct llama_model * model);
int32_t llama_model_n_embd(const struct llama_model * model);
int32_t llama_model_n_layer(const struct llama_model * model);
int32_t llama_model_n_head(const struct llama_model * model);
int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);
int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);
const char * llama_model_chat_template(const struct llama_model * model, const char * name);
uint64_t llama_model_n_params(const struct llama_model * model);
uint64_t llama_model_size(const struct llama_model * model);
bool llama_model_has_encoder(const struct llama_model * model);
llama_token llama_model_decoder_start_token(const struct llama_model * model);

int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special
);

int32_t llama_detokenize(
    const struct llama_vocab * vocab,
    const llama_token * tokens,
    int32_t n_tokens,
    char * text,
    int32_t text_len_max,
    bool remove_special,
    bool unparse_special
);

int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special
);

int32_t llama_chat_apply_template(
    const char * tmpl,
    const llama_chat_message * chat,
    size_t n_msg,
    bool add_ass,
    char * buf,
    int32_t length
);

bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);
llama_token llama_vocab_bos(const struct llama_vocab * vocab);

llama_batch llama_batch_get_one(llama_token * tokens, int32_t n_tokens);
llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void llama_batch_free(llama_batch batch);
int32_t llama_encode(struct llama_context * ctx, llama_batch batch);
int32_t llama_decode(struct llama_context * ctx, llama_batch batch);

void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);
void llama_set_warmup(struct llama_context * ctx, bool warmup);
void llama_synchronize(struct llama_context * ctx);
float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

struct llama_sampler * llama_sampler_chain_init(llama_sampler_chain_params params);
void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl);
struct llama_sampler * llama_sampler_init_greedy(void);
struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
struct llama_sampler * llama_sampler_init_top_k(int32_t k);
struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);
struct llama_sampler * llama_sampler_init_min_p(float p, size_t min_keep);
struct llama_sampler * llama_sampler_init_temp(float t);
llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);
void llama_sampler_accept(struct llama_sampler * smpl, llama_token token);
void llama_sampler_free(struct llama_sampler * smpl);

struct llama_perf_context_data llama_perf_context(const struct llama_context * ctx);
void llama_perf_context_reset(struct llama_context * ctx);
struct llama_perf_sampler_data llama_perf_sampler(const struct llama_sampler * chain);
void llama_perf_sampler_reset(struct llama_sampler * chain);

size_t llama_state_get_size(struct llama_context * ctx);
size_t llama_state_get_data(struct llama_context * ctx, uint8_t * dst, size_t size);
size_t llama_state_set_data(struct llama_context * ctx, const uint8_t * src, size_t size);
CDEF;
    }
}
