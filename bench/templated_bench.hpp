#pragma once

#include "bench_utils.hpp"
#include "layer_creator.hpp"
#include <chrono>
#include <RTNeural.h>

#if MODELT_AVAILABLE

float runTemplatedBench(const std::vector<vec_type>& signal, const size_t n_samples,
    const std::string& layer_type, size_t in_size, size_t out_size)
{
    using namespace RTNeural;
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<float>;

    auto run_layer = [=, &signal] (auto& layer) -> float
    {
        auto start = clock_t::now();
        for(size_t i = 0; i < n_samples; ++i)
            layer.forward(signal[i].data());
        return std::chrono::duration_cast<second_t>(clock_t::now() - start).count();
    };

    float duration = 10000.0;
    if(layer_type == "dense")
    {
        if(in_size == 8 && out_size == 8)
        {
            ModelT<float, 8, 8, DenseT<float, 8, 8>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 12 && out_size == 12)
        {
            ModelT<float, 12, 12, DenseT<float, 12, 12>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<float, 16, 16, DenseT<float, 16, 16>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 24 && out_size == 24)
        {
            ModelT<float, 24, 24, DenseT<float, 24, 24>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 1)
        {
            ModelT<float, 8, 1, DenseT<float, 8, 1>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 12 && out_size == 1)
        {
            ModelT<float, 12, 1, DenseT<float, 12, 1>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 1)
        {
            ModelT<float, 16, 1, DenseT<float, 16, 1>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 24 && out_size == 1)
        {
            ModelT<float, 24, 1, DenseT<float, 24, 1>> model;
            randomise_dense (model.get<0>());
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "gru")
    {
        if(in_size == 1 && out_size == 8)
        {
            ModelT<float, 1, 8, GRULayerT<float, 1, 8>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 1 && out_size == 12)
        {
            ModelT<float, 1, 12, GRULayerT<float, 1, 12>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 1 && out_size == 16)
        {
            ModelT<float, 1, 16, GRULayerT<float, 1, 16>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 1 && out_size == 24)
        {
            ModelT<float, 1, 16, GRULayerT<float, 1, 16>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<float, 8, 8, GRULayerT<float, 8, 8>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 12 && out_size == 12)
        {
            ModelT<float, 12, 12, GRULayerT<float, 12, 12>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<float, 16, 16, GRULayerT<float, 16, 16>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 24 && out_size == 24)
        {
            ModelT<float, 24, 24, GRULayerT<float, 24, 24>> model;
            randomise_gru (model.get<0>());
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "lstm")
    {
        if(in_size == 1 && out_size == 8)
        {
            ModelT<float, 1, 8, LSTMLayerT<float, 1, 8>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 1 && out_size == 12)
        {
            ModelT<float, 1, 12, LSTMLayerT<float, 1, 12>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 1 && out_size == 16)
        {
            ModelT<float, 1, 16, LSTMLayerT<float, 1, 16>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 1 && out_size == 24)
        {
            ModelT<float, 1, 24, LSTMLayerT<float, 1, 24>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<float, 8, 8, LSTMLayerT<float, 8, 8>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 12 && out_size == 12)
        {
            ModelT<float, 12, 12, LSTMLayerT<float, 12, 12>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<float, 16, 16, LSTMLayerT<float, 16, 16>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else if(in_size == 24 && out_size == 24)
        {
            ModelT<float, 24, 24, LSTMLayerT<float, 24, 24>> model;
            randomise_lstm (model.get<0>());
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "conv1d")
    {
        if(in_size == 4 && out_size == 4)
        {
            constexpr size_t kernel_size = 3; // in_size - 1
            ModelT<float, 4, 4, Conv1DT<float, 4, 4, kernel_size, 1>> model;
            randomise_conv1d (model.get<0>(), kernel_size);
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            constexpr size_t kernel_size = 7; // in_size - 1
            constexpr size_t dilation_rate = 2; // in_size / 4
            ModelT<float, 8, 8, Conv1DT<float, 8, 8, kernel_size, dilation_rate>> model;
            randomise_conv1d (model.get<0>(), kernel_size);
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            constexpr size_t kernel_size = 15; // in_size - 1
            constexpr size_t dilation_rate = 4; // in_size / 4
            ModelT<float, 16, 16, Conv1DT<float, 16, 16, kernel_size, dilation_rate>> model;
            randomise_conv1d (model.get<0>(), kernel_size);
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "tanh")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<float, 4, 4, TanhActivationT<float, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<float, 8, 8, TanhActivationT<float, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<float, 16, 16, TanhActivationT<float, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "fast_tanh")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<float, 4, 4, FastTanhT<float, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<float, 8, 8, FastTanhT<float, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<float, 16, 16, FastTanhT<float, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "relu")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<float, 4, 4, ReLuActivationT<float, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<float, 8, 8, ReLuActivationT<float, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<float, 16, 16, ReLuActivationT<float, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "sigmoid")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<float, 4, 4, SigmoidActivationT<float, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<float, 8, 8, SigmoidActivationT<float, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<float, 16, 16, SigmoidActivationT<float, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }
    else if(layer_type == "softmax")
    {
        if(in_size == 4 && out_size == 4)
        {
            ModelT<float, 4, 4, SoftmaxActivationT<float, 4>> model;
            duration = run_layer(model);
        }
        else if(in_size == 8 && out_size == 8)
        {
            ModelT<float, 8, 8, SoftmaxActivationT<float, 8>> model;
            duration = run_layer(model);
        }
        else if(in_size == 16 && out_size == 16)
        {
            ModelT<float, 16, 16, SoftmaxActivationT<float, 16>> model;
            duration = run_layer(model);
        }
        else
        {
            std::cout << "Layer size not supported for templated benchmarks!" << std::endl;
        }
    }

    return duration;
}

#endif // MODELT_AVAILABLE
