"""ATen inference BatchNorm for the official FX2AIT NHWC vision lowering.

All arithmetic uses FP32, with epsilon inside sqrt and an explicit output cast.
The pinned upstream converter constructs a default-FP16 epsilon and places it
outside sqrt; these are frontend semantic issues, not BF16 device capability.
"""
from aitemplate.compiler import ops
from aitemplate.compiler.public import Tensor,FuncEnum

def batch_norm(target,args,kwargs,name):
 x,weight,bias,mean,var,training,momentum,eps,cudnn=args
 if training:raise ValueError('DNN1 baseline only supports inference BatchNorm')
 dtype=x.dtype()
 def f32(x):return ops.cast()(x,'float32') if x.dtype()!='float32' else x
 mu=f32(mean);variance=f32(var);gamma=f32(weight);beta=f32(bias)
 epsilon=Tensor(shape=[],dtype='float32',value=float(eps))
 denom=ops.elementwise(FuncEnum.SQRT)(ops.elementwise(FuncEnum.ADD)(variance,epsilon))
 centered=ops.elementwise(FuncEnum.SUB)(f32(x),mu)
 normalized=ops.elementwise(FuncEnum.DIV)(centered,denom)
 out=ops.elementwise(FuncEnum.ADD)(ops.elementwise(FuncEnum.MUL)(normalized,gamma),beta)
 return ops.cast()(out,dtype) if dtype!='float32' else out

def hardtanh(target,args,kwargs,name):
 x=args[0];lo=args[1] if len(args)>1 else -1;hi=args[2] if len(args)>2 else 1
 out=x
 if lo is not None:out=ops.elementwise(FuncEnum.MAX)(out,Tensor(shape=[],dtype=x.dtype(),value=lo))
 if hi is not None:out=ops.elementwise(FuncEnum.MIN)(out,Tensor(shape=[],dtype=x.dtype(),value=hi))
 return out
