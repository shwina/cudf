/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/sorting.hpp>

#include <numeric>

#include <thrust/transform_scan.h>

namespace cudf {
namespace detail {

std::unique_ptr<column> index_of(
  table_view const& input,
  table_view const& keys,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  rmm::device_vector<size_type> idx(input.num_rows());
  thrust::sequence(rmm::exec_policy(stream)->on(stream), idx.begin(), idx.end(), 0);

  column_view idx_view(
    data_type(type_to_id<size_type>()), idx.size(), idx.data().get(), nullptr, 0);
  table_view key_table({table_view({idx_view}), input});

  auto sorted_key_table = sort_by_key(key_table, input);

  std::vector<size_type> column_selection(input.num_columns());
  std::iota(column_selection.begin(), column_selection.end(), 1);

  auto index_column = sorted_key_table->view().column(0);
  auto search_table = sorted_key_table->view().select(column_selection);

  auto lb = lower_bound(search_table,
                        keys,
                        std::vector<order>(search_table.num_columns(), order::ASCENDING),
                        std::vector<null_order>(search_table.num_columns(), null_order::AFTER),
                        mr,
                        stream);

  auto ub = upper_bound(search_table,
                        keys,
                        std::vector<order>(search_table.num_columns(), order::ASCENDING),
                        std::vector<null_order>(search_table.num_columns(), null_order::AFTER),
                        mr,
                        stream);

  // calculate size of the result
  rmm::device_vector<size_type> offsets(lb->size() + 1, 0);
  thrust::transform_inclusive_scan(
    thrust::device,
    thrust::make_zip_iterator(
      thrust::make_tuple(lb->view().begin<size_type>(), ub->view().begin<size_type>())),
    thrust::make_zip_iterator(
      thrust::make_tuple(lb->view().end<size_type>(), ub->view().end<size_type>())),
    offsets.begin() + 1,
    [] __device__(thrust::tuple<size_type, size_type> elem) -> size_type {
      return thrust::get<1>(elem) - thrust::get<0>(elem);
    },
    thrust::plus<size_type>());

  auto result_size = (size_type)offsets[lb->size()];

  auto result = cudf::make_numeric_column(
    data_type(type_to_id<size_type>()), result_size, mask_state::UNALLOCATED, stream, mr);

  auto d_result = result->mutable_view().begin<size_type>();
  auto d_idx    = index_column.begin<size_type>();

  thrust::for_each_n(
    rmm::exec_policy()->on(stream),
    thrust::make_zip_iterator(thrust::make_tuple(
      lb->view().begin<size_type>(), ub->view().begin<size_type>(), offsets.begin())),
    result_size,
    [d_idx, d_result] __device__(thrust::tuple<size_type, size_type, size_type> index_tup) {
      thrust::for_each(thrust::seq,
                       thrust::make_counting_iterator(thrust::get<0>(index_tup)),
                       thrust::make_counting_iterator(thrust::get<1>(index_tup)),
                       [d_idx, d_result, index_tup] __device__(size_type i) {
                         *(d_result + thrust::get<2>(index_tup)) = *(d_idx + i);
                       });
    });
  return result;
}

}  // namespace detail

std::unique_ptr<column> index_of(table_view const& input,
                                 table_view const& keys,
                                 rmm::mr::device_memory_resource* mr)
{
  return detail::index_of(input, keys, mr, 0);
}

}  // namespace cudf
