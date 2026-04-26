namespace DotNetFoundryLLM.ModelFormats.Gguf;

/// <summary>Describes a single tensor stored in a GGUF file.</summary>
public sealed class GgufTensorInfo
{
    /// <summary>Tensor name as stored in the GGUF file.</summary>
    public required string Name { get; init; }

    /// <summary>
    /// Dimensions in GGML column-major order (ne[0] is the fastest-varying axis).
    /// </summary>
    public required ulong[] Dims { get; init; }

    /// <summary>GGML data type of the tensor elements.</summary>
    public required GgufTensorType TensorType { get; init; }

    /// <summary>
    /// Byte offset of the tensor data from the start of the data segment
    /// (i.e., relative to the aligned data start, not the file start).
    /// </summary>
    public required ulong Offset { get; init; }

    /// <summary>Total number of elements in the tensor (product of all dimensions).</summary>
    public long ElementCount
    {
        get
        {
            long count = 1;
            foreach (var d in Dims)
            {
                count *= (long)d;
            }

            return count;
        }
    }

    /// <summary>Number of dimensions (rank) of the tensor.</summary>
    public int Rank => Dims.Length;
}
