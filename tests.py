from e2cnn import gspaces
from e2cnn import nn as nn2


for kernel_size in [5, 7]:
    print("\nKernel size: {}".format(kernel_size))
    for N in range(2, 17):
        print("\nRotation Angles: {}".format(N))
        r2_act_rot = gspaces.Rot2dOnR2(N=N)

        in_type_rot = nn2.FieldType(r2_act_rot, [r2_act_rot.trivial_repr])
        out_type1_rot = nn2.FieldType(r2_act_rot, [r2_act_rot.regular_repr])
        out_type2_rot = out_type1_rot
        C1 = nn2.R2Conv(in_type_rot, out_type1_rot, kernel_size=kernel_size)
        C2 = nn2.R2Conv(out_type1_rot, out_type2_rot, kernel_size=kernel_size)
        print(C1.weights.shape)
        print(C2.weights.shape)


for kernel_size in [5, 7]:
    print("\nKernel size: {}".format(kernel_size))
    for N in range(2, 17):
        print("\nRotation Angles: {}".format(N))
        r2_act_fliprot = gspaces.FlipRot2dOnR2(N=N)

        in_type_fliprot = nn2.FieldType(r2_act_fliprot, [r2_act_fliprot.trivial_repr])
        out_type1_fliprot = nn2.FieldType(r2_act_fliprot, [r2_act_fliprot.regular_repr])
        out_type2_fliprot = out_type1_fliprot
        C1 = nn2.R2Conv(in_type_fliprot, out_type1_fliprot, kernel_size=kernel_size)
        C2 = nn2.R2Conv(out_type1_fliprot, out_type2_fliprot, kernel_size=kernel_size)
        print(C1.weights.shape)
        print(C2.weights.shape)
    #r2_act_fliprot = gspaces.FlipRot2dOnR2(N=N)
