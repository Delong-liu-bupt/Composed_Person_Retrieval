"""
Compatibility wrapper for different transformers versions
This module handles the import changes in transformers library
"""

# Try to import from different locations based on transformers version
try:
    # Old location (transformers < 4.30)
    from transformers.modeling_utils import apply_chunking_to_forward
except ImportError:
    try:
        # New location (transformers >= 4.30)
        from transformers.pytorch_utils import apply_chunking_to_forward
    except ImportError:
        # Fallback: define a compatible version locally
        def apply_chunking_to_forward(
            forward_fn,
            chunk_size,
            chunk_dim,
            *input_tensors,
        ):
            """
            This function chunks the input_tensors into smaller input tensor parts of size chunk_size over the
            dimension chunk_dim. It then applies a layer forward_fn to each chunk independently to save memory.

            If chunk_size is -1, it will apply the forward_fn to the whole tensor.
            """

            assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)

            # No chunking needed
            if chunk_size == -1:
                return forward_fn(*input_tensors)

            # Chunk and apply forward
            num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
            if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
                num_chunks += 1

            output_chunks = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, input_tensors[0].shape[chunk_dim])

                chunk_slice = [slice(None)] * input_tensors[0].ndim
                chunk_slice[chunk_dim] = slice(start_idx, end_idx)

                input_tensors_chunk = [tensor[chunk_slice] for tensor in input_tensors]
                output_chunk = forward_fn(*input_tensors_chunk)
                output_chunks.append(output_chunk)

            import torch
            return torch.cat(output_chunks, dim=chunk_dim)

# Export the function
__all__ = ['apply_chunking_to_forward']