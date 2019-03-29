#include "types.hpp"
#include "copying/utilities/copying_utils.cuh"
#include "copying/utilities/copying_utils.hpp"
#include "utilities/type_dispatcher.hpp"
#include "utilities/error_utils.hpp"

namespace cudf {

namespace {

using cudf::utilities::bitmask_partition_params;
using cudf::utilities::data_partition_params;
using cudf::utilities::block_type;
using cudf::utilities::double_block_type;
using cudf::utilities::BLOCK_MASK_VALUE;
using cudf::utilities::BITS_PER_BLOCK;

__device__ __forceinline__
void calculate_data_params(data_partition_params* params,
                           gdf_size_type const    input_size,
                           gdf_index_type const*  indexes,
                           gdf_size_type const    indexes_size,
                           gdf_index_type const   indexes_position) {
  params->input_offset = gdf_index_type{0};
  if (indexes_position != 0) {
    params->input_offset = indexes[indexes_position - 1];
  }
  if (indexes_size <= indexes_position) {
    params->row_size = input_size - params->input_offset;
  }
  else {
    params->row_size = indexes[indexes_position] - params->input_offset;
  }
}

__device__ __forceinline__
void calculate_bitmask_params(bitmask_partition_params* params,
                              gdf_valid_type*           output_bitmask,
                              gdf_valid_type const*     input_bitmask,
                              gdf_size_type const       input_size,
                              gdf_index_type const*     indexes,
                              gdf_size_type const       indexes_size,
                              gdf_index_type const      indexes_position) {
  params->block_output = reinterpret_cast<block_type*>(output_bitmask);
  params->block_input = reinterpret_cast<block_type const*>(input_bitmask);
  
  //
  gdf_index_type input_index_begin{0};
  if (indexes_position != 0) {
    input_index_begin = indexes[indexes_position - 1];
  }
  gdf_index_type input_index_end = input_size;
  if (indexes_position < indexes_size) {
    input_index_end = indexes[indexes_position];
  }
 
  params->input_offset = input_index_begin / BITS_PER_BLOCK;
  params->rotate_input = input_index_begin % BITS_PER_BLOCK;
  params->mask_last = (double_block_type{1} << ((input_index_end - input_index_begin) % BITS_PER_BLOCK)) - double_block_type{1};

  params->input_block_length = (input_size + (BITS_PER_BLOCK - 1)) / BITS_PER_BLOCK;
  params->partition_block_length = ((input_index_end - input_index_begin) + (BITS_PER_BLOCK - 1)) / BITS_PER_BLOCK;
}

template <typename ColumnType>
__global__
void split_data_kernel(ColumnType*           output_data,
                       ColumnType const*     input_data,
                       gdf_size_type const   input_size,
                       gdf_index_type const* indexes,
                       gdf_size_type const   indexes_size,
                       gdf_index_type const  indexes_position) {
  // Obtain the indexes for copying
  cudf::utilities::data_partition_params data_params;
  calculate_data_params(&data_params, input_size, indexes, indexes_size, indexes_position);

  // Perform the copy operation
  cudf::utilities::copy_data<ColumnType>(&data_params, output_data, input_data);
}

__global__
void split_bitmask_kernel(gdf_valid_type*       output_bitmask,
                          gdf_valid_type const* input_bitmask,
                          gdf_size_type const   input_size,
                          gdf_index_type const* indexes,
                          gdf_size_type const   indexes_size,
                          gdf_index_type const  indexes_position) {
  // Gather the indexes for copying
  cudf::utilities::bitmask_partition_params bitmask_params;
  calculate_bitmask_params(&bitmask_params,
                           output_bitmask,
                           input_bitmask,
                           input_size,
                           indexes,
                           indexes_size,
                           indexes_position);


  // Calculate kernel parameters
  gdf_size_type row_index = threadIdx.x + blockIdx.x * blockDim.x;
  gdf_size_type row_step = blockDim.x * gridDim.x;

  // Perform the copying operation
  while (row_index < bitmask_params.partition_block_length) {
    cudf::utilities::copy_bitmask(&bitmask_params, row_index);
    row_index += row_step;
  }
}

using cudf::utilities::BaseCopying;
class Split : protected BaseCopying {
public:
  Split(gdf_column const*   input_column,
        gdf_column const*   indexes,
        cudf::column_array* output_columns,
        cudaStream_t*       streams,
        gdf_size_type       streams_size)
  : BaseCopying(input_column, indexes, output_columns, streams, streams_size)
  { }

public:
  template <typename ColumnType>
  void operator()() {
    // Perform operation
    gdf_size_type columns_quantity = output_columns_->num_columns();
    for (gdf_index_type index = 0; index < columns_quantity; ++index) {
      // Obtain the column
      gdf_column* output_column = output_columns_->get_column(index);

      // Empty output column
      if (output_column->size == 0) {
        continue;
      }

      // Obtain the stream
      cudaStream_t stream = get_stream(index);

      // Calculate kernel occupancy for data
      auto kernel_data_occupancy = calculate_kernel_data_occupancy(output_column->size);

      // Make a copy of the data in the gdf_column
      split_data_kernel<ColumnType>
      <<<
        kernel_data_occupancy.grid_size,
        kernel_data_occupancy.block_size,
        cudf::utilities::NO_DYNAMIC_MEMORY,
        stream
      >>>(
        reinterpret_cast<ColumnType*>(output_column->data),
        reinterpret_cast<ColumnType const*>(input_column_->data),
        input_column_->size,
        reinterpret_cast<gdf_index_type const*>(indexes_->data),
        indexes_->size,
        index
      );

      // Calculate kernel occupancy for bitmask
      auto kernel_bitmask_occupancy = calculate_kernel_bitmask_occupancy(output_column->size);
      
      // Make a copy of the bitmask in the gdf_column
      split_bitmask_kernel
      <<<
        kernel_bitmask_occupancy.grid_size,
        kernel_bitmask_occupancy.block_size,
        cudf::utilities::NO_DYNAMIC_MEMORY,
        stream
      >>>(
        output_column->valid,
        input_column_->valid,
        input_column_->size,
        reinterpret_cast<gdf_index_type const*>(indexes_->data),
        indexes_->size,
        index
      );

      CHECK_STREAM(stream);
    }
  }
};

} // namespace


namespace detail {

void split(gdf_column const*   input_column,
           gdf_column const*   indexes,
           cudf::column_array* output_columns,
           cudaStream_t*       streams,
           gdf_size_type       streams_size) {
  Split split(input_column, indexes, output_columns, streams, streams_size);
  cudf::type_dispatcher(input_column->dtype, split);
}

} // namespace detail
} // namespace cudf


namespace cudf {

void split(gdf_column const*   input_column,
           gdf_column const*   indexes,
           cudf::column_array* output_columns) {
  cudf::detail::split(input_column, indexes, output_columns, nullptr, 0);
}

} // namespace cudf
