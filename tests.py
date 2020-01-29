from e2cnn import gspaces
from e2cnn import nn as nn2

import math

"""
for kernel_size in [5, 7]:
    print("\nKernel size: {}".format(kernel_size))
    for N in range(1, 17):
        print("\nRotation Angles: {}".format(N))
        if N == 1:
            r2_act_rot = gspaces.TrivialOnR2()
        else:
            r2_act_rot = gspaces.Rot2dOnR2(N=N)

        scaling_factor = 4/(math.sqrt(N))
        scaling_factor*= math.sqrt(2)
        print(scaling_factor)
        in_type_rot = nn2.FieldType(r2_act_rot, round(scaling_factor*32)*[r2_act_rot.trivial_repr])
        out_type1_rot = nn2.FieldType(r2_act_rot, round(scaling_factor*16)*[r2_act_rot.regular_repr])
        out_type2_rot = nn2.FieldType(r2_act_rot, round(scaling_factor*8)*[r2_act_rot.regular_repr])
        C1 = nn2.R2Conv(in_type_rot, out_type1_rot, kernel_size=kernel_size)
        C2 = nn2.R2Conv(out_type1_rot, out_type2_rot, kernel_size=kernel_size)
        print(C1.weights.shape)
        print(C2.weights.shape)
"""


for kernel_size in [5, 7]:
    print("\nKernel size: {}".format(kernel_size))
    for N in range(1, 17):
        print("\nRotation Angles: {}".format(N))
        if N == 1:
            r2_act_fliprot = gspaces.TrivialOnR2()
        else:
            r2_act_fliprot = gspaces.FlipRot2dOnR2(N=N)
        scaling_factor = 4/(math.sqrt(N))
        print(scaling_factor)
        in_type_fliprot = nn2.FieldType(r2_act_fliprot, round(scaling_factor*32)*[r2_act_fliprot.trivial_repr])
        out_type1_fliprot = nn2.FieldType(r2_act_fliprot, round(scaling_factor*16)*[r2_act_fliprot.regular_repr])
        out_type2_fliprot = nn2.FieldType(r2_act_fliprot, round(scaling_factor*8)*[r2_act_fliprot.regular_repr])
        C1 = nn2.R2Conv(in_type_fliprot, out_type1_fliprot, kernel_size=kernel_size)
        C2 = nn2.R2Conv(out_type1_fliprot, out_type2_fliprot, kernel_size=kernel_size)
        print(C1.weights.shape)
        print(C2.weights.shape)
