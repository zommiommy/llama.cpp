#include "sampling.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <unordered_map>

// Tests that common_sampler_sample_and_accept_n_rejection preserves the target
// distribution regardless of what the draft distribution Q looks like, for a
// single draft token and for a block of draft tokens.

static int test_speculative_rejection_sampling(common_params & params, int n_trials) {
    params.sampling.backend_sampling = false;
    auto llama_init = common_init_from_params(params);

    auto * model_tgt = llama_init->model();
    auto * ctx_tgt   = llama_init->context();
    if (!model_tgt || !ctx_tgt) {
        LOG_ERR("failed to initialize target model and context\n");
        return 1;
    }

    const llama_tokens prompt = { 1, 2, 3, 1, 2 };

    // Process the prompt with the target model to get its true distribution for
    // the position right after the prompt.
    common_batch batch = common_batch_get_one(ctx_tgt, prompt);
    if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("failed to decode the test prompt\n");
        return 1;
    }

    common_params_sampling sampling;
    sampling.seed     = 18;
    sampling.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    // the test model's logits are nearly flat; a sharp target tells it apart from the random drafts, so a verifier
    // that lets draft tokens through fails
    sampling.temp     = 0.05f;

    // Initialize the sampler for the target model.
    common_sampler_ptr smpl(common_sampler_init(model_tgt, sampling));
    if (!smpl) {
        LOG_ERR("failed to initialize target sampler\n");
        return 1;
    }

    // Sample once so the sampler chain runs and populates cur_p. We don't need
    // the returned token id itself so the return value is ignored.
    common_sampler_sample(smpl.get(), ctx_tgt, -1);

    // Get the target candidates distribution in sorted order.
    const llama_token_data_array * dist_tgt = common_sampler_get_candidates(smpl.get(), true);

    // Clone smpl before the trial loop starts. We need the dist_tgt pointer to stay valid.
    common_sampler_ptr smpl_verify(common_sampler_clone(smpl.get()));
    if (!smpl_verify) {
        LOG_ERR("failed to clone target sampler\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::mt19937 rng(18);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> q(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        q[i].id = i;
        q[i].logit = 0.0f;
    }

    // a random draft distribution and a token drawn from it
    auto draw_draft = [&](llama_token & tok) {
        float q_sum = 0.0f;

        for (auto & td : q) {
            td.p = uni(rng);
            q_sum += td.p;
        }

        const float inv_sum = 1.0f / q_sum;
        for (auto & td : q) {
            td.p *= inv_sum;
        }

        float u = uni(rng);

        // Fallback to the last bucket to absorb residual probability from FP precision loss
        tok = q.back().id;
        for (const auto & e : q) {
            u -= e.p;
            if (u <= 0.0f) {
                tok = e.id;
                break;
            }
        }
        return q;
    };

    bool ok = true;

    // Every index reuses the prompt's last output logits, so the target draws every token from the same
    // distribution and the first two tokens of the output must each follow it. With one draft token, the bonus draw
    // of the second index is stale and only the first output token is checked.
    for (const int n_draft : { 1, 3 }) {
        const int n_check = n_draft == 1 ? 1 : 2;

        std::vector<std::unordered_map<llama_token, int>> token_counts(n_check);

        for (int trial = 0; trial < n_trials; ++trial) {
            std::vector<llama_token> out;
            while ((int) out.size() < n_check) {
                llama_tokens draft(n_draft);
                std::vector<std::vector<llama_token_data>> draft_q(n_draft);
                for (int i = 0; i < n_draft; ++i) {
                    draft_q[i] = draw_draft(draft[i]);
                }
                const std::vector<int> idxs(n_draft + 1, -1);

                const std::vector<llama_token> result =
                    common_sampler_sample_and_accept_n_rejection(smpl_verify.get(), ctx_tgt, idxs, draft, draft_q, false);
                out.insert(out.end(), result.begin(), result.end());
            }
            for (int k = 0; k < n_check; ++k) {
                token_counts[k][out[k]]++;
            }
        }

        for (int k = 0; k < n_check; ++k) {
            LOG("\n");
            LOG("%d draft tokens, output token %d: empirical vs target distribution after %d trials:\n", n_draft, k, n_trials);

            size_t covered = 0;

            auto get_count_for = [&](llama_token id) -> int {
                const auto it = token_counts[k].find(id);
                return it != token_counts[k].end() ? it->second : 0;
            };

            // Iterate over the target distribution
            for (size_t i = 0; i < dist_tgt->size; ++i) {
                const llama_token id  = dist_tgt->data[i].id;
                const float       p_t = dist_tgt->data[i].p;

                const int   count = get_count_for(id);
                const float p_e   = (float) count / n_trials;
                const float diff  = std::fabs(p_t - p_e);

                // We only deal with ids in dist_tgt, and if each trial result was one
                // of dist_tgt's candidates then covered ends up equal to the number of
                // trials. If not this inconsistency tells us how many trials produced
                // a token outside of the target's distribution.
                covered += count;

                const float standard_error = std::sqrt(p_t * (1.0f - p_t) / n_trials);
                const float tolerance      = 6.0f * standard_error;

                LOG("token %3d : target = %f, empirical = %f, diff = %f, tol = %f\n", id, p_t, p_e, diff, tolerance);

                if (diff > tolerance) {
                    LOG_ERR("%d draft tokens, output token %d: token %d: empirical probability %f too far from target %f (diff %f > tol %f)\n",
                            n_draft, k, id, p_e, p_t, diff, tolerance);
                    ok = false;
                }
            }

            if (covered != (size_t) n_trials) {
                LOG_ERR("%zu of %d trials produced a token outside dist_tgt's candidates\n",
                        (size_t) n_trials - covered, n_trials);
                ok = false;
            }
        }
    }

    if (!ok) {
        return 1;
    }

    return 0;
}

struct parsed_args {
    int n_trials;
    std::vector<char *> argv;
};

static std::optional<parsed_args> parse_n_trials_arg(int argc, char ** argv, int default_n_trials) {
    parsed_args result;
    result.n_trials = default_n_trials;
    result.argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n-trials") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --n-trials requires a number argument\n", __func__);
                return std::nullopt;
            }
            int value = std::atoi(argv[i + 1]);
            if (value <= 0) {
                LOG_ERR("%s: --n-trials requires a positive number argument\n", __func__);
                return std::nullopt;
            }
            result.n_trials = value;
            i++;
        } else {
            result.argv.push_back(argv[i]);
        }
    }
    result.argv.push_back(nullptr);

    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 256;

    common_init();

    auto parsed = parse_n_trials_arg(argc, argv, 5000);
    if (!parsed) {
        return 1;
    }
    const int fargc = (int) parsed->argv.size() - 1;

    if (!common_params_parse(fargc, parsed->argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    LOG("Using target model '%s'\n", params.model.path.c_str());

    llama_backend_init();

    const int result = test_speculative_rejection_sampling(params, parsed->n_trials);
    if (result == 0) {
        LOG("\nAll tests passed.\n");
    } else {
        LOG("\nSome tests failed.\n");
    }

    llama_backend_free();
    return result;
}
