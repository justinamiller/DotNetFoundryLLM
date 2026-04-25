using System.Buffers.Binary;
using System.Text;
using DotNetFoundryLLM.ModelFormats.Gguf;
using Xunit;

namespace DotNetFoundryLLM.ModelFormats.Gguf.Tests;

/// <summary>Tests for <see cref="GgufReader"/>.</summary>
public sealed class GgufReaderTests
{
    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>Builds a minimal valid GGUF v3 byte array with no tensors or metadata.</summary>
    private static byte[] BuildMinimalGguf(uint version = 3, ulong tensorCount = 0, ulong kvCount = 0)
    {
        var ms = new System.IO.MemoryStream();
        using var w = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        w.Write((byte)'G');
        w.Write((byte)'G');
        w.Write((byte)'U');
        w.Write((byte)'F');
        w.Write(version);           // version
        w.Write(tensorCount);       // tensor_count
        w.Write(kvCount);           // metadata_kv_count

        return ms.ToArray();
    }

    private static void WriteGgufString(System.IO.BinaryWriter w, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    // ── Magic validation ──────────────────────────────────────────────────────

    [Fact]
    public void Parse_InvalidMagic_Throws()
    {
        var data = new byte[] { 0x00, 0x01, 0x02, 0x03, 0x03, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        var ex = Assert.Throws<Core.ModelLoadException>(() => GgufReader.Parse(data));
        Assert.Contains("magic", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(0u)]
    [InlineData(1u)]
    [InlineData(4u)]
    public void Parse_InvalidVersion_Throws(uint version)
    {
        var data = BuildMinimalGguf(version);
        var ex = Assert.Throws<Core.ModelLoadException>(() => GgufReader.Parse(data));
        Assert.Contains("version", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    // ── Empty file ────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(2u)]
    [InlineData(3u)]
    public void Parse_EmptyGguf_SucceedsWithNoTensors(uint version)
    {
        var data = BuildMinimalGguf(version);
        var gguf = GgufReader.Parse(data);

        Assert.Equal(version, gguf.Version);
        Assert.Empty(gguf.Tensors);
        Assert.Empty(gguf.Metadata);
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    [Fact]
    public void Parse_StringMetadata_Reads()
    {
        var ms = new System.IO.MemoryStream();
        using var w = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        w.Write((byte)'G'); w.Write((byte)'G'); w.Write((byte)'U'); w.Write((byte)'F');
        w.Write(3u);           // version
        w.Write(0UL);          // tensor_count
        w.Write(1UL);          // kv_count

        // KV: key="general.architecture", type=String, value="llama"
        WriteGgufString(w, "general.architecture");
        w.Write((uint)GgufValueType.String);
        WriteGgufString(w, "llama");

        var gguf = GgufReader.Parse(ms.ToArray());

        Assert.True(gguf.Metadata.ContainsKey("general.architecture"));
        Assert.Equal("llama", gguf.Metadata["general.architecture"].StringValue);
    }

    [Fact]
    public void Parse_Uint32Metadata_Reads()
    {
        var ms = new System.IO.MemoryStream();
        using var w = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        w.Write((byte)'G'); w.Write((byte)'G'); w.Write((byte)'U'); w.Write((byte)'F');
        w.Write(3u);
        w.Write(0UL);
        w.Write(1UL);

        WriteGgufString(w, "llama.block_count");
        w.Write((uint)GgufValueType.Uint32);
        w.Write(32u);

        var gguf = GgufReader.Parse(ms.ToArray());

        Assert.Equal(32u, gguf.Metadata["llama.block_count"].Uint32Value);
    }

    [Fact]
    public void Parse_ArrayOfStrings_Reads()
    {
        var ms = new System.IO.MemoryStream();
        using var w = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        w.Write((byte)'G'); w.Write((byte)'G'); w.Write((byte)'U'); w.Write((byte)'F');
        w.Write(3u);
        w.Write(0UL);
        w.Write(1UL);

        WriteGgufString(w, "tokenizer.ggml.tokens");
        w.Write((uint)GgufValueType.Array);
        w.Write((uint)GgufValueType.String); // element type
        w.Write(2UL);                        // count (uint64 in v3)
        WriteGgufString(w, "hello");
        WriteGgufString(w, "world");

        var gguf = GgufReader.Parse(ms.ToArray());

        var arr = gguf.Metadata["tokenizer.ggml.tokens"];
        Assert.Equal(GgufValueType.Array, arr.ValueType);
        Assert.Equal(2, arr.ArrayValue!.Count);
        Assert.Equal("hello", arr.ArrayValue[0].StringValue);
        Assert.Equal("world", arr.ArrayValue[1].StringValue);
    }

    // ── Tensor info ───────────────────────────────────────────────────────────

    [Fact]
    public void Parse_TensorInfo_ReadsNameTypeOffset()
    {
        var ms = new System.IO.MemoryStream();
        using var w = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        w.Write((byte)'G'); w.Write((byte)'G'); w.Write((byte)'U'); w.Write((byte)'F');
        w.Write(3u);
        w.Write(1UL); // 1 tensor
        w.Write(0UL); // 0 kv

        WriteGgufString(w, "output_norm.weight");
        w.Write(1u);           // n_dims
        w.Write(4096UL);       // dim[0]
        w.Write((uint)GgufTensorType.F32);
        w.Write(0UL);          // offset

        var gguf = GgufReader.Parse(ms.ToArray());

        Assert.Single(gguf.Tensors);
        var t = gguf.Tensors[0];
        Assert.Equal("output_norm.weight", t.Name);
        Assert.Equal(GgufTensorType.F32, t.TensorType);
        Assert.Equal(4096L, t.ElementCount);
        Assert.Equal(0UL,   t.Offset);
    }

    // ── FindTensor ────────────────────────────────────────────────────────────

    [Fact]
    public void FindTensor_ExistingName_ReturnsTensorInfo()
    {
        var ms = new System.IO.MemoryStream();
        using var w = new System.IO.BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);

        w.Write((byte)'G'); w.Write((byte)'G'); w.Write((byte)'U'); w.Write((byte)'F');
        w.Write(3u);
        w.Write(1UL);
        w.Write(0UL);

        WriteGgufString(w, "token_embd.weight");
        w.Write(2u);
        w.Write(32000UL); // vocab_size
        w.Write(4096UL);  // hidden_size
        w.Write((uint)GgufTensorType.F32);
        w.Write(0UL);

        var gguf = GgufReader.Parse(ms.ToArray());

        var info = gguf.FindTensor("token_embd.weight");
        Assert.NotNull(info);
        Assert.Equal(2, info.Rank);
        Assert.Equal(32000UL, info.Dims[0]);
        Assert.Equal(4096UL,  info.Dims[1]);
    }

    [Fact]
    public void FindTensor_MissingName_ReturnsNull()
    {
        var gguf = GgufReader.Parse(BuildMinimalGguf());
        Assert.Null(gguf.FindTensor("nonexistent"));
    }
}
