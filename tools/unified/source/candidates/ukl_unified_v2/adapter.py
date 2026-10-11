from candidates.large_graph_fast_v1.adapter import Adapter

class UKLAdapter(Adapter):
    dataset='UKL'

    def run_window(self,batches,epoch,phase):
        if self.device.type!='cuda':return super().run_window(batches,epoch,phase)
        from .runtime.pressure import observe,interval,settle
        if not hasattr(self,'pressure_windows'):
            self.pressure_windows=[];self.pressure_gates={}
        if phase not in self.pressure_gates:
            self.pressure_gates[phase]=settle(self.r.check,self.r.event)
        before=observe()
        result=super().run_window(batches,epoch,phase)
        value=interval(before,observe());value.update(epoch=epoch+1,phase=phase)
        self.pressure_windows.append(value)
        self.r.event('memory_pressure_window',**value)
        if phase=='training' and not value['passed']:
            raise RuntimeError('Measured window affected by current memory pressure')
        return result

    def finish(self):
        result=super().finish()
        if self.device.type=='cuda':
            result['memory_pressure']=dict(gates=self.pressure_gates,windows=self.pressure_windows,
                passed=all(v['passed'] for v in self.pressure_windows if v['phase']=='training'))
        return result
