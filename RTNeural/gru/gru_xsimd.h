#ifndef GRUXSIMD_H_INCLUDED
#define GRUXSIMD_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include <vector>
namespace RTNeural
{

/**
 * Dynamic implementation of a gated recurrent unit (GRU) layer
 * with tanh activation and sigmoid recurrent activation.
 * 
 * To ensure that the recurrent state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 */
template <typename T>
class GRULayer : public Layer<T>
{
public:
    /** Constructs a GRU layer for a given input and output size. */
    GRULayer(int in_size, int out_size);
    GRULayer(std::initializer_list<int> sizes);
    GRULayer(const GRULayer& other);
    GRULayer& operator=(const GRULayer& other);
    virtual ~GRULayer();

    /** Resets the state of the GRU. */
    void reset() override { std::fill(ht1.begin(), ht1.end(), (T)0); }

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "gru"; }

    /** Performs forward propagation for this layer. */
    virtual inline void forward(const T* input, T* h) override
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
        {
            zVec[i] = vMult(zWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size) + vMult(zWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
            rVec[i] = vMult(rWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size) + vMult(rWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
            cVec[i] = vMult(cWeights.W[i].data(), input, prod_in.data(), Layer<T>::in_size);
            cTmp[i] = vMult(cWeights.U[i].data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
        }

        vAdd(zVec.data(), zWeights.b[0].data(), zVec.data(), Layer<T>::out_size);
        vAdd(zVec.data(), zWeights.b[1].data(), zVec.data(), Layer<T>::out_size);
        sigmoid(zVec.data(), zVec.data(), Layer<T>::out_size);

        vAdd(rVec.data(), rWeights.b[0].data(), rVec.data(), Layer<T>::out_size);
        vAdd(rVec.data(), rWeights.b[1].data(), rVec.data(), Layer<T>::out_size);
        sigmoid(rVec.data(), rVec.data(), Layer<T>::out_size);

        vAdd(cTmp.data(), cWeights.b[1].data(), cTmp.data(), Layer<T>::out_size);
        vProd(cTmp.data(), rVec.data(), cTmp.data(), Layer<T>::out_size);
        vAdd(cTmp.data(), cVec.data(), cVec.data(), Layer<T>::out_size);
        vAdd(cVec.data(), cWeights.b[0].data(), cVec.data(), Layer<T>::out_size);
        tanh(cVec.data(), cVec.data(), Layer<T>::out_size);

        vSub(ones.data(), zVec.data(), h, Layer<T>::out_size);
        vProd(h, cVec.data(), h, Layer<T>::out_size);
        vProd(zVec.data(), ht1.data(), prod_out.data(), Layer<T>::out_size);
        vAdd(h, prod_out.data(), h, Layer<T>::out_size);

        vCopy(h, ht1.data(), Layer<T>::out_size);
    }

    /**
     * Sets the layer kernel weights.
     * 
     * The weights vector must have size weights[in_size][3 * out_size]
     */
    void setWVals(T** wVals);

    /**
     * Sets the layer recurrent weights.
     * 
     * The weights vector must have size weights[out_size][3 * out_size]
     */
    void setUVals(T** uVals);

    /**
     * Sets the layer bias.
     * 
     * The bias vector must have size weights[2][3 * out_size]
     */
    void setBVals(T** bVals);

    /**
     * Sets the layer kernel weights.
     * 
     * The weights vector must have size weights[in_size][3 * out_size]
     */
    void setWVals(const std::vector<std::vector<T>>& wVals);

    /**
     * Sets the layer recurrent weights.
     * 
     * The weights vector must have size weights[out_size][3 * out_size]
     */
    void setUVals(const std::vector<std::vector<T>>& uVals);

    /**
     * Sets the layer bias.
     * 
     * The bias vector must have size weights[2][3 * out_size]
     */
    void setBVals(const std::vector<std::vector<T>>& bVals);

    /** Returns the kernel weight for the given indices. */
    T getWVal(int i, int k) const noexcept;

    /** Returns the recurrent weight for the given indices. */
    T getUVal(int i, int k) const noexcept;

    /** Returns the bias value for the given indices. */
    T getBVal(int i, int k) const noexcept;

protected:
    using vec_type = std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)>;
    using vec2_type = std::vector<vec_type>;

    vec_type ht1;

    struct WeightSet
    {
        WeightSet(int in_size, int out_size);
        ~WeightSet();

        vec2_type W; // kernel weights
        vec2_type U; // recurrent weights
        vec_type b[2]; // bias
        const int out_size;
    };

    WeightSet zWeights;
    WeightSet rWeights;
    WeightSet cWeights;

    vec_type zVec;
    vec_type rVec;
    vec_type cVec;
    vec_type cTmp;

    vec_type prod_in;
    vec_type prod_out;
    vec_type ones;
};

//====================================================
/**
 * Static implementation of a gated recurrent unit (GRU) layer
 * with tanh activation and sigmoid recurrent activation.
 * 
 * To ensure that the recurrent state is initialized to zero,
 * please make sure to call `reset()` before your first call to
 * the `forward()` method.
 */
template <typename T, int in_sizet, int out_sizet>
class GRULayerT
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_in_size = ceil_div(in_sizet, v_size);
    static constexpr auto v_out_size = ceil_div(out_sizet, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    GRULayerT();

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "gru"; }

    /** Returns false since GRU is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Resets the state of the GRU. */
    void reset();

    /** Performs forward propagation for this layer. */
    template <int N = in_size>
    inline typename std::enable_if<(N > 1), void>::type
    forward(const v_type (&ins)[v_in_size])
    {
        // compute zt
        recurrent_mat_mul(outs, Uz, zt);
        kernel_mat_mul(ins, Wz, kernel_outs);
        for(int i = 0; i < v_out_size; ++i)
            zt[i] = sigmoid(zt[i] + bz[i] + kernel_outs[i]);

        // compute rt
        recurrent_mat_mul(outs, Ur, rt);
        kernel_mat_mul(ins, Wr, kernel_outs);
        for(int i = 0; i < v_out_size; ++i)
            rt[i] = sigmoid(rt[i] + br[i] + kernel_outs[i]);

        // compute h_hat
        recurrent_mat_mul(outs, Uh, ct);
        kernel_mat_mul(ins, Wh, kernel_outs);
        for(int i = 0; i < v_out_size; ++i)
            ht[i] = xsimd::tanh(xsimd::fma(rt[i], ct[i] + bh1[i], bh0[i] + kernel_outs[i]));

        // compute output
        for(int i = 0; i < v_out_size; ++i)
            outs[i] = xsimd::fma((v_type((T)1.0) - zt[i]), ht[i], zt[i] * outs[i]);
    }

    /** Performs forward propagation for this layer. */
    template <int N = in_size>
    inline typename std::enable_if<N == 1, void>::type
    forward(const v_type (&ins)[v_in_size])
    {
        // compute zt
        recurrent_mat_mul(outs, Uz, zt);
        for(int i = 0; i < v_out_size; ++i)
            zt[i] = sigmoid(xsimd::fma(Wz_1[i], ins[0], zt[i] + bz[i]));

        // compute rt
        recurrent_mat_mul(outs, Ur, rt);
        for(int i = 0; i < v_out_size; ++i)
            rt[i] = sigmoid(xsimd::fma(Wr_1[i], ins[0], rt[i] + br[i]));

        // compute h_hat
        recurrent_mat_mul(outs, Uh, ct);
        for(int i = 0; i < v_out_size; ++i)
            ht[i] = xsimd::tanh(xsimd::fma(rt[i], ct[i] + bh1[i], xsimd::fma(Wh_1[i], ins[0], bh0[i])));

        // compute output
        for(int i = 0; i < v_out_size; ++i)
            outs[i] = xsimd::fma((v_type((T)1.0) - zt[i]), ht[i], zt[i] * outs[i]);
    }

    /**
     * Sets the layer kernel weights.
     * 
     * The weights vector must have size weights[in_size][3 * out_size]
     */
    void setWVals(const std::vector<std::vector<T>>& wVals);

    /**
     * Sets the layer recurrent weights.
     * 
     * The weights vector must have size weights[out_size][3 * out_size]
     */
    void setUVals(const std::vector<std::vector<T>>& uVals);

    /**
     * Sets the layer bias.
     * 
     * The bias vector must have size weights[2][3 * out_size]
     */
    void setBVals(const std::vector<std::vector<T>>& bVals);

    v_type outs[v_out_size];

private:
    static inline void recurrent_mat_mul(const v_type (&vec)[v_out_size], const v_type (&mat)[out_size][v_out_size], v_type (&out)[v_out_size]) noexcept
    {
        T sums alignas(16)[out_size] { (T)0 };
        for(int i = 0; i < v_size; ++i)
        {
            for(int j = 0; j < v_out_size; ++j)
            {
                for(int k = 0; k < v_out_size; ++k)
                    sums[i + j * v_size] += xsimd::hadd(mat[i + j * v_size][k] * vec[k]);
            }
        }

        for(int i = 0; i < v_out_size; ++i)
            out[i] = xsimd::load_aligned(sums + i * v_size);
    }

    static inline void kernel_mat_mul(const v_type (&vec)[v_in_size], const v_type (&mat)[out_size][v_in_size], v_type (&out)[v_out_size]) noexcept
    {
        T sums alignas(16)[out_size] { (T)0 };
        for(int i = 0; i < v_size; ++i)
        {
            for(int j = 0; j < v_out_size; ++j)
            {
                for(int k = 0; k < v_in_size; ++k)
                    sums[i + j * v_size] += xsimd::hadd(mat[i + j * v_size][k] * vec[k]);
            }
        }

        for(int i = 0; i < v_out_size; ++i)
            out[i] = xsimd::load_aligned(sums + i * v_size);
    }

    static inline v_type sigmoid(v_type x) noexcept
    {
        return (T)1.0 / ((T)1.0 + xsimd::exp(-x));
    }

    // kernel weights
    v_type Wz[out_size][v_in_size];
    v_type Wr[out_size][v_in_size];
    v_type Wh[out_size][v_in_size];
    v_type kernel_outs[v_out_size];

    // single-input kernel weights
    v_type Wz_1[v_out_size];
    v_type Wr_1[v_out_size];
    v_type Wh_1[v_out_size];

    // recurrent weights
    v_type Uz[out_size][v_out_size];
    v_type Ur[out_size][v_out_size];
    v_type Uh[out_size][v_out_size];

    // biases
    v_type bz[v_out_size];
    v_type br[v_out_size];
    v_type bh0[v_out_size];
    v_type bh1[v_out_size];

    // intermediate vars
    v_type zt[v_out_size];
    v_type rt[v_out_size];
    v_type ct[v_out_size];
    v_type ht[v_out_size];
};

} // namespace RTNeural

#endif // GRUXSIMD_H_INCLUDED
