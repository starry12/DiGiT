"""Exact FP32 tensor division and first-order gradients on genuine CUDA."""
from .common import *


def main():
    require(not (OUT/'native_checks.json').exists(),'Preserve native evidence')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Use physical GPU2')
    import subprocess
    fields=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
    require(fields[0].strip()==GPU_UUID and int(fields[1])<1024,'GPU busy or changed')
    setup()
    import torch
    import DiGiTMeanNormalizeCUDA as native
    from .normalization import MeanNormalize, divide
    require(Path(native.__file__).resolve()==NORMALIZATION_BINARY and native.MEAN_NORMALIZE_API==1,'Wrong mean extension')
    torch.set_num_threads(1)
    torch.manual_seed(913)
    cases=[]
    def exact(a,b):
        require(torch.equal(torch.isnan(a),torch.isnan(b)),'NaN mask changed')
        mask=~torch.isnan(a)
        require(torch.equal(a.contiguous().view(torch.int32)[mask],b.contiguous().view(torch.int32)[mask]),'FP32 bits changed')
    for dtype in (torch.int32,torch.int64):
        for rows,cols in ((0,128),(1,0),(1,1),(17,7),(33,128),(1024,128),(907,128)):
            for layout in ('contiguous','transpose','expanded'):
                maximum=2**31-1 if dtype==torch.int32 else 2**40
                choices=torch.tensor([-7,0,1,2,3,7,127,255,65537,2**24-1,2**24+1,maximum],dtype=dtype,device='cuda')
                degrees=choices.repeat((rows*2+11)//12)[:rows*2:2]
                if layout=='contiguous':x=torch.randn(rows,cols,device='cuda')
                elif layout=='transpose':x=torch.randn(cols,rows,device='cuda').t()
                else:x=torch.randn(1,cols,device='cuda').expand(rows,cols)
                x=x.detach().requires_grad_()
                denominator=degrees.clamp(1,maximum).float().reshape(rows,1)
                expected=x/denominator
                got=MeanNormalize.apply(x,degrees,maximum)
                exact(got,expected)
                grad=torch.randn(cols,rows,device='cuda').t()
                actual_grad=torch.autograd.grad(got,x,grad)[0]
                expected_grad=torch.autograd.grad(expected,x,grad)[0]
                exact(actual_grad,expected_grad)
                # Expanded upstream gradient exercises zero strides in backward.
                exact(divide(torch.ones(1,cols,device='cuda').expand(rows,cols),degrees,maximum,True),
                    torch.ones(rows,cols,device='cuda')/denominator)
                cases.append(dict(rows=rows,columns=cols,degree_dtype=str(dtype),layout=layout,forward_backward_bit_exact=True))
    values=torch.tensor([0.,-0.,1.,-1.,1.17549435e-38,1.40129846e-45,3.4028235e38,float('inf'),-float('inf'),float('nan')],device='cuda')
    for dtype in (torch.int32,torch.int64):
        degrees=torch.tensor([0,1,3,7,128,2**24+1],device='cuda',dtype=dtype)
        x=values.expand(len(degrees),-1)
        exact(divide(x,degrees,2**31-1),x/degrees.clamp(1,2**31-1).float()[:,None])
        cases.append(dict(degree_dtype=str(dtype),special_values=True,finite_and_infinity_bits_exact=True,nan_masks_equal=True))
    # Producer and consumer both use the requested non-default stream.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x=torch.randn(37,128,device='cuda',requires_grad=True);d=torch.arange(37,device='cuda',dtype=torch.int32)
        y=MeanNormalize.apply(x,d,37);g=torch.autograd.grad(y.sum(),x)[0]
        expected=x/d.clamp(1,37).float()[:,None]
    torch.cuda.current_stream().wait_stream(stream)
    exact(y,expected);exact(g,torch.ones_like(x)/d.clamp(1,37).float()[:,None])
    invalid=0
    for shape in ((-1,128,128,1,1,1,32),(1,128,128,1,1,0,32),(1,128,128,1,1,1,8)):
        try:native.forward(0,0,0,*shape,0)
        except (ValueError,RuntimeError):invalid+=1
        else:raise RuntimeError('Invalid native contract accepted')
    for bad_x,bad_d,maximum in ((torch.ones(2,3),torch.ones(2,dtype=torch.int32),3),
            (torch.ones(2,3,device='cuda',dtype=torch.float64),torch.ones(2,device='cuda',dtype=torch.int32),3),
            (torch.ones(2,3,device='cuda'),torch.ones(3,device='cuda',dtype=torch.int32),3)):
        try:divide(bad_x,bad_d,maximum)
        except ValueError:invalid+=1
        else:raise RuntimeError('Invalid tensor contract accepted')
    torch.cuda.synchronize()
    write(OUT/'native_checks.json',dict(passed=True,cases=cases,case_count=len(cases),
        first_order_backward_bit_exact=True,non_default_stream_passed=True,invalid_cases_rejected=invalid,
        binary_sha256=sha(NORMALIZATION_BINARY),tested_sha256={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in ('normalization.py','native_tests.py','native/mean_normalize.cu')},
        raw_ssd_access=False,performance_evidence=False,torch_version=torch.__version__))
    print('Passed mean native forward/backward:',len(cases),'cases and stream/invalid checks',flush=True)


if __name__=='__main__':main()
