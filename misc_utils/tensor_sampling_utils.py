'''
Utilities that are useful to sub- or up-sample weights tensors.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np

def sample_tensors(weights_list, sampling_instructions, axes=None, init=None, mean=0.0, stddev=0.005):
    '''
    Can sub-sample and/or up-sample individual dimensions of the tensors in the given list
    of input tensors.

    It is possible to sub-sample some dimensions and up-sample other dimensions at the same time.

    The tensors in the list will be sampled consistently, i.e. for any given dimension that
    corresponds among all tensors in the list, the same elements will be picked for every tensor
    along that dimension.

    For dimensions that are being sub-sampled, you can either provide a list of the indices
    that should be picked, or you can provide the number of elements to be sub-sampled, in which
    case the elements will be chosen at random.

    For dimensions that are being up-sampled, "filler" elements will be insterted at random
    positions along the respective dimension. These filler elements will be initialized either
    with zero or from a normal distribution with selectable mean and standard deviation.

    Arguments:
        weights_list (list): A list of Numpy arrays. Each array represents one of the tensors
            to be sampled. The tensor with the greatest number of dimensions must be the first
            element in the list. For example, in the case of the weights of a 2D convolutional
            layer, the kernel must be the first element in the list and the bias the second,
            not the other way around. For all tensors in the list after the first tensor, the
            lengths of each of their axes must identical to the length of some axis of the
            first tensor.
        sampling_instructions (list): A list that contains the sampling instructions for each
            dimension of the first tensor. If the first tensor has `n` dimensions, then this
            must be a list of length `n`. That means, sampling instructions for every dimension
            of the first tensor must still be given even if not all dimensions should be changed.
            The elements of this list can be either lists of integers or integers. If the sampling
            instruction for a given dimension is a list of integers, then these integers represent
            the indices of the elements of that dimension that will be sub-sampled. If the sampling
            instruction for a given dimension is an integer, then that number of elements will be
            sampled along said dimension. If the integer is greater than the number of elements
            of the input tensors in that dimension, that dimension will be up-sampled. If the integer
            is smaller than the number of elements of the input tensors in that dimension, that
            dimension will be sub-sampled. If the integer is equal to the number of elements
            of the input tensors in that dimension, that dimension will remain the same.
        axes (list, optional): Only relevant if `weights_list` contains more than one tensor.
            This list contains a list for each additional tensor in `weights_list` beyond the first.
            Each of these lists contains integers that determine to which axes of the first tensor
            the axes of the respective tensor correspond. For example, let the first tensor be a
            4D tensor and the second tensor in the list be a 2D tensor. If the first element of
            `axis` is the list `[2,3]`, then that means that the two axes of the second tensor
            correspond to the last two axes of the first tensor, in the same order. The point of
            this list is for the program to know, if a given dimension of the first tensor is to
            be sub- or up-sampled, which dimensions of the other tensors in the list must be
            sub- or up-sampled accordingly.
        init (list, optional): Only relevant for up-sampling. Must be `None` or a list of strings
            that determines for each tensor in `weights_list` how the newly inserted values should
            be initialized. The possible values are 'gaussian' for initialization from a normal
            distribution with the selected mean and standard deviation (see the following two arguments),
            or 'zeros' for zero-initialization. If `None`, all initializations default to
            'gaussian'.
        mean (float, optional): Only relevant for up-sampling. The mean of the values that will
            be inserted into the tensors at random in the case of up-sampling.
        stddev (float, optional): Only relevant for up-sampling. The standard deviation of the
            values that will be inserted into the tensors at random in the case of up-sampling.

    Returns:
        A list containing the sampled tensors in the same order in which they were given.
    '''

    first_tensor = weights_list[0]

    if (not isinstance(sampling_instructions, (list, tuple))) or (len(sampling_instructions) != first_tensor.ndim):
        raise ValueError("The sampling instructions must be a list whose length is the number of dimensions of the first tensor in `weights_list`.")

    if (not init is None) and len(init) != len(weights_list):
        raise ValueError("`init` must either be `None` or a list of strings that has the same length as `weights_list`.")

    up_sample = [] # Store the dimensions along which we need to up-sample.
    out_shape = [] # Store the shape of the output tensor here.
    # Store two stages of the new (sub-sampled and/or up-sampled) weights tensors in the following two lists.
    subsampled_weights_list = [] # Tensors after sub-sampling, but before up-sampling (if any).
    upsampled_weights_list = [] # Sub-sampled tensors after up-sampling (if any), i.e. final output tensors.

    # Create the slicing arrays from the sampling instructions.
    sampling_slices = []
    for i, sampling_inst in enumerate(sampling_instructions):
        if isinstance(sampling_inst, (list, tuple)):
            amax = np.amax(np.array(sampling_inst))
            if amax >= first_tensor.shape[i]:
                raise ValueError("The sample instructions for dimension {} contain index {}, which is greater than the length of that dimension.".format(i, amax))
            sampling_slices.append(np.array(sampling_inst))
            out_shape.append(len(sampling_inst))
        elif isinstance(sampling_inst, int):
            out_shape.append(sampling_inst)
            if sampling_inst == first_tensor.shape[i]:
                # Nothing to sample here, we're keeping the original number of elements along this axis.
                sampling_slice = np.arange(sampling_inst)
                sampling_slices.append(sampling_slice)
            elif sampling_inst < first_tensor.shape[i]:
                # We want to SUB-sample this dimension. Randomly pick `sample_inst` many elements from it.
                sampling_slice1 = np.array([0]) # We will always sample class 0, the background class.
                # Sample the rest of the classes.
                sampling_slice2 = np.sort(np.random.choice(np.arange(1, first_tensor.shape[i]), sampling_inst - 1, replace=False))
                sampling_slice = np.concatenate([sampling_slice1, sampling_slice2])
                sampling_slices.append(sampling_slice)
            else:
                # We want to UP-sample. Pick all elements from this dimension.
                sampling_slice = np.arange(first_tensor.shape[i])
                sampling_slices.append(sampling_slice)
                up_sample.append(i)
        else:
            raise ValueError("Each element of the sampling instructions must be either an integer or a list/tuple of integers, but received `{}`".format(type(sampling_inst)))

    # Process the first tensor.
    subsampled_first_tensor = np.copy(first_tensor[np.ix_(*sampling_slices)])
    subsampled_weights_list.append(subsampled_first_tensor)

    # Process the other tensors.
    if len(weights_list) > 1:
        for j in range(1, len(weights_list)):
            this_sampling_slices = [sampling_slices[i] for i in axes[j-1]] # Get the sampling slices for this tensor.
            subsampled_weights_list.append(np.copy(weights_list[j][np.ix_(*this_sampling_slices)]))

    if up_sample:
        # Take care of the dimensions that are to be up-sampled.

        out_shape = np.array(out_shape)

        # Process the first tensor.
        if init is None or init[0] == 'gaussian':
            upsampled_first_tensor = np.random.normal(loc=mean, scale=stddev, size=out_shape)
        elif init[0] == 'zeros':
            upsampled_first_tensor = np.zeros(out_shape)
        else:
            raise ValueError("Valid initializations are 'gaussian' and 'zeros', but received '{}'.".format(init[0]))
        # Pick the indices of the elements in `upsampled_first_tensor` that should be occupied by `subsampled_first_tensor`.
        up_sample_slices = [np.arange(k) for k in subsampled_first_tensor.shape]
        for i in up_sample:
            # Randomly select across which indices of this dimension to scatter the elements of `new_weights_tensor` in this dimension.
            up_sample_slice1 = np.array([0])
            up_sample_slice2 = np.sort(np.random.choice(np.arange(1, upsampled_first_tensor.shape[i]), subsampled_first_tensor.shape[i] - 1, replace=False))
            up_sample_slices[i] = np.concatenate([up_sample_slice1, up_sample_slice2])
        upsampled_first_tensor[np.ix_(*up_sample_slices)] = subsampled_first_tensor
        upsampled_weights_list.append(upsampled_first_tensor)

        # Process the other tensors
        if len(weights_list) > 1:
            for j in range(1, len(weights_list)):
                if init is None or init[j] == 'gaussian':
                    upsampled_tensor = np.random.normal(loc=mean, scale=stddev, size=out_shape[axes[j-1]])
                elif init[j] == 'zeros':
                    upsampled_tensor = np.zeros(out_shape[axes[j-1]])
                else:
                    raise ValueError("Valid initializations are 'gaussian' and 'zeros', but received '{}'.".format(init[j]))
                this_up_sample_slices = [up_sample_slices[i] for i in axes[j-1]] # Get the up-sampling slices for this tensor.
                upsampled_tensor[np.ix_(*this_up_sample_slices)] = subsampled_weights_list[j]
                upsampled_weights_list.append(upsampled_tensor)

        return upsampled_weights_list
    else:
        return subsampled_weights_list
