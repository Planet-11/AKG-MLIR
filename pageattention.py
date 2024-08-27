
import torch
import torch.nn.functional as F

def paged_attention_torch(
    query,  # [num_seqs, num_query_heads, head_size]
    key_cache,  # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    context_lens,  # [num_seqs]
    scale,  # float32
    num_seqs,  # int
    num_heads,  # int
    cache_block_stride,  # int
    MAX_CONTEXT_LEN,  # int
    BLOCK_SIZE,  # int
    HEAD_SIZE,  # int, must be power of 2
):


    # Initialize output tensor
    output = torch.zeros((num_seqs, num_heads, HEAD_SIZE), device=query.device, dtype=query.dtype)

    for seq_idx in range(num_seqs):
        for head_idx in range(num_heads):
            query_head = query[seq_idx, head_idx]
            context_len = context_lens[seq_idx]

            keys = torch.zeros((context_len, HEAD_SIZE), device=query.device, dtype=query.dtype)
            values = torch.zeros((context_len, HEAD_SIZE), device=query.device, dtype=query.dtype)

            for tok_idx in range(context_len):
                logical_block_idx = tok_idx // BLOCK_SIZE
                physical_block_idx = block_tables[seq_idx, logical_block_idx]

                start_of_block_offset = physical_block_idx * cache_block_stride + head_idx * HEAD_SIZE * BLOCK_SIZE
                tok_idx_within_block = tok_idx % BLOCK_SIZE
                tok_offsets = start_of_block_offset + BLOCK_SIZE * torch.arange(0, HEAD_SIZE, device=query.device) + tok_idx_within_block

                tok_key = key_cache.view(-1)[tok_offsets]
                tok_value = value_cache.view(-1)[tok_offsets]

                keys[tok_idx] = tok_key
                values[tok_idx] = tok_value

            # Scale the query
            query_head = query_head * scale

            # Compute attention scores
            scores = query_head @ keys.transpose(-2, -1)
            
            # Mask scores
            scores_masked = torch.full((MAX_CONTEXT_LEN,), float('-inf'), device=query.device, dtype=query.dtype)
            scores_masked[:context_len] = scores
            # print(scores_masked[:context_len],'context_len',context_len)

            # Compute softmax of scores
            logits = F.softmax(scores_masked[:context_len], dim=0)
            print(logits)

            # Compute weighted values
            weighted_values = torch.sum(values[:context_len] * logits[:, None], dim=0)

            # Store the result in the output tensor
            output[seq_idx, head_idx] = weighted_values

    return output


# Expect block table to map
# logical bid (block id) -> (physical bid, # filled)
# In tests, it maps: logical bid -> physical bid


class TestPagedAttentionTorch():
  
    def test_paged_attention_torch(self):
        # Define the parameters for the test
        num_seqs = 2
        num_query_heads = 2
        num_kv_heads = 2
        head_size = 64  # Assuming head_size is a power of 2
        block_size = 64
        num_blocks = 4
        max_num_blocks_per_seq = 2
        context_lens = torch.tensor([128, 64])
        scale = 1.0
        cache_block_stride = 64  # Assuming this is the correct stride
        MAX_CONTEXT_LEN = 128

        # Initialize the tensors with random values
        query = torch.randn(num_seqs, num_query_heads, head_size)
        key_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size)
        value_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size)
        block_tables = torch.randint(0, num_blocks, (num_seqs, max_num_blocks_per_seq))

        # Run the paged_attention_torch function
        output = paged_attention_torch(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            scale=scale,
            num_seqs=num_seqs,
            num_heads=num_query_heads,
            cache_block_stride=cache_block_stride,
            MAX_CONTEXT_LEN=MAX_CONTEXT_LEN,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
        )

        # Verify the output shape
        expected_shape = (num_seqs, num_query_heads, head_size)
        # self.assertEqual(output.shape, expected_shape)
        #print(output)
        # print(score_masked[:context_])
        # Verify the output values (this might be tricky without knowing the expected output)
        # You might want to add some assertions here to check the correctness of the values
        # For example, check if the output contains NaNs or infs
        # self.assertFalse(torch.isnan(output).any())
        # self.assertFalse(torch.isinf(output).any())

if __name__ == "__main__":
    TestPagedAttentionTorch().test_paged_attention_torch()

