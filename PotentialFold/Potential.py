import numpy as np
# from scipy.interpolate import UnivariateSpline # Keep if used elsewhere, not directly for this class
# from scipy.interpolate import interp1d # Not strictly needed for import here, but assumed to be used in Cubic.py
import os # Keep if used elsewhere
from torch.autograd import Function
import torch
import math # Keep, math.pi might be used if you adapt torsion/angle
import json # Keep if used elsewhere

# batched_index_select is not used in this revised version for interp1d,
# as interp1d objects are directly callable.
# def batched_index_select(input, dim, index):
#     # ...

class cubic_batch_dis_class(Function): # Consider renaming to reflect it's no longer strictly cubic
    @staticmethod
    def forward(ctx, input1, list_of_interp1d_callables, min_dis_ignored=None, max_dis_ignored=None, bin_num_ignored=None):
        """
        Evaluates a list of interp1d callable functions on a tensor of input distances.

        Args:
            ctx: Context for saving variables for backward pass.
            input1 (torch.Tensor): A 1D tensor of distance values to evaluate.
            list_of_interp1d_callables (list): A list of callable scipy.interpolate.interp1d
                                               objects. The length of this list MUST match
                                               the number of elements in input1 if input1 is not empty.
            min_dis_ignored, max_dis_ignored, bin_num_ignored: These parameters are
                                                                kept for API compatibility
                                                                but are not used.
        Returns:
            torch.Tensor: A tensor of evaluated spline values, same shape as input1.
        """
        # Type and length checks
        if not isinstance(list_of_interp1d_callables, list):
            # Allow for an empty list if input1 is also empty
            if not (input1.numel() == 0 and not list_of_interp1d_callables):
                 raise TypeError(
                    "`list_of_interp1d_callables` must be a Python list of callables. "
                    f"Got {type(list_of_interp1d_callables)}"
                )

        if input1.numel() != 0 and len(list_of_interp1d_callables) == 0:
            raise ValueError(
                f"Received {input1.numel()} input distances but an empty list of spline callables."
            )
        
        if input1.numel() != 0 and input1.numel() != len(list_of_interp1d_callables):
            raise ValueError(
                f"Mismatch in number of input distances ({input1.numel()}) "
                f"and number of spline callables ({len(list_of_interp1d_callables)})."
            )

        # Handle empty input tensor
        if input1.numel() == 0:
            ctx.save_for_backward(input1.clone()) # Save a clone for safety
            ctx.callable_splines_list_for_backward = []
            return torch.empty_like(input1)

        # Proceed with computation for non-empty inputs
        input_detached_cpu = input1.detach().cpu().numpy()
        output_np = np.empty_like(input_detached_cpu, dtype=np.float64) # Ensure float64 for scipy

        for i in range(len(input_detached_cpu)):
            spline_callable = list_of_interp1d_callables[i]
            # Ensure the input to the spline is a scalar float, not a 0-dim array
            scalar_input = input_detached_cpu[i].item() if hasattr(input_detached_cpu[i], 'item') else input_detached_cpu[i]
            try:
                output_np[i] = spline_callable(scalar_input)
            except Exception as e:
                # Add more context to the error if a spline call fails
                print(f"Error evaluating spline {i} with input {scalar_input}: {e}")
                # Potentially raise or handle by setting a default value like np.nan or a large penalty
                # For now, re-raising to make the error visible.
                raise e


        output_tensor = torch.tensor(output_np, dtype=input1.dtype, device=input1.device)
        
        ctx.save_for_backward(input1.clone()) # Save a clone
        ctx.callable_splines_list_for_backward = list_of_interp1d_callables

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Grad for input1, then None for the other args of forward()
        # (list_of_interp1d_callables, min_dis_ignored, max_dis_ignored, bin_num_ignored)
        
        # Check if gradient for input1 is actually needed
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None

        input1_original_props_clone, = ctx.saved_tensors
        callable_splines_list = ctx.callable_splines_list_for_backward

        if input1_original_props_clone.numel() == 0:
            return torch.zeros_like(input1_original_props_clone), None, None, None, None
        
        # Ensure callable_splines_list is not empty if input is not empty
        if not callable_splines_list and input1_original_props_clone.numel() > 0:
             # This case should ideally be caught by checks in forward, but as a safeguard:
            print("Warning: callable_splines_list_for_backward is empty in backward pass but input was not.")
            return torch.zeros_like(input1_original_props_clone), None, None, None, None

        input1_detached_cpu = input1_original_props_clone.detach().cpu().numpy()
        grad_input1_np = np.empty_like(input1_detached_cpu, dtype=np.float64) # Ensure float64 for scipy
        
        epsilon = 1e-7 # Adjusted epsilon for potentially more stable numerical differentiation

        for i in range(len(input1_detached_cpu)):
            dist_val = input1_detached_cpu[i].item() if hasattr(input1_detached_cpu[i], 'item') else input1_detached_cpu[i]
            spline_callable = callable_splines_list[i]
            
            try:
                val_plus_eps = spline_callable(dist_val + epsilon)
                val_minus_eps = spline_callable(dist_val - epsilon)
            except Exception as e:
                print(f"Error during numerical differentiation for spline {i} with input {dist_val}: {e}")
                # Handle error, e.g., by setting derivative to 0 or NaN
                grad_input1_np[i] = 0.0 # Or np.nan, or re-raise
                continue # Move to the next iteration


            derivative = (val_plus_eps - val_minus_eps) / (2 * epsilon)
            grad_input1_np[i] = derivative
            
        grad_input1_tensor = torch.tensor(grad_input1_np, dtype=grad_output.dtype, device=grad_output.device)
        
        final_grad_input1 = grad_output * grad_input1_tensor
        
        return final_grad_input1, None, None, None, None

# Wrapper function
def cubic_distance(input1, list_of_interp1d_callables, min_dis=None, max_dis=None, bin_num=None):
    return cubic_batch_dis_class.apply(input1, list_of_interp1d_callables, min_dis, max_dis, bin_num)


# --- Keep or adapt other functions/classes below as needed ---

def LJpotential(dis,th):
    r = ( (th+0.5) / (dis+0.5))**6 # Added small epsilon to denominator to prevent division by zero if dis = -0.5
    return (r**2 - 2*r)

# --- Placeholder for torsion and angle classes ---
# If you are also changing these to use interp1d, they would need similar adaptations.
# For now, I'm keeping their original structure with placeholders for forward/backward
# to ensure the file is syntactically correct if these are called.

class cubic_batch_torsion_class(Function):
    @staticmethod
    def forward(ctx,input1,coe,x,num_bin):
        # Placeholder - adapt if changing torsion potentials
        print("Warning: cubic_batch_torsion_class.forward is a placeholder.")
        ctx.save_for_backward(input1, coe, x) 
        return torch.zeros_like(input1) 
    @staticmethod
    def backward(ctx,grad_output):
        print("Warning: cubic_batch_torsion_class.backward is a placeholder.")
        input1, coe, x = ctx.saved_tensors
        return torch.zeros_like(input1), None, None, None

def cubic_torsion(input1,coe,x,num_bin):
    return cubic_batch_torsion_class.apply(input1,coe,x,num_bin)


class cubic_batch_angle_class(Function):
    @staticmethod
    def forward(ctx,input1,coe,x,num_bin=12):
        print("Warning: cubic_batch_angle_class.forward is a placeholder.")
        ctx.save_for_backward(input1, coe, x)
        return torch.zeros_like(input1)
    @staticmethod
    def backward(ctx,grad_output):
        print("Warning: cubic_batch_angle_class.backward is a placeholder.")
        input1, coe, x = ctx.saved_tensors
        return torch.zeros_like(input1), None, None, None

def cubic_angle(input1,coe,x,num_bin):
    return cubic_batch_angle_class.apply(input1,coe,x,num_bin)