from habitat.analysis import SPECIAL_OPERATIONS
from habitat.analysis.arguments import Arguments
from habitat.analysis.operation import MeasuredOperation
from habitat.analysis.trace import Trace
from habitat.profiling.operation import OperationProfiler
from habitat.tracking.base import TrackerBase
from habitat.tracking.callable import CallableTracker


class OperationTracker(TrackerBase):
    def __init__(self, device, metrics=None, metrics_threshold_ms=0):
        super().__init__()
        self._device = device
        self._callable_tracker = CallableTracker(self._hook_creator)
        self._profiler = OperationProfiler(
            device,
            metrics,
            metrics_threshold_ms,
        )
        self._processing_hook = False

        self._operations = []

    def start_tracking(self):
        super().start_tracking()
        self._callable_tracker.start_tracking()

    def stop_tracking(self):
        super().stop_tracking()
        self._callable_tracker.stop_tracking()

    def get_tracked_trace(self):
        return Trace(self._device, self._operations)

    def _hook_creator(self, func):
        def hook(*args, **kwargs):
            # NOTE: We use self._processing_hook to handle cases where we have
            #       hooks on nested function calls.
            if self._processing_hook:
                return func(*args, **kwargs)

            self._processing_hook = True
            try:
                # We only track the arguments if the operation is "special"
                # (i.e. we use special handling to scale it to a different
                # device).
                print(f"TEST: ----------------> {func.__name__}")
                is_special_op = func.__name__ in SPECIAL_OPERATIONS
                arguments = (
                    Arguments.from_raw_arguments(args, kwargs)
                    if is_special_op else None
                )

                if (func.__name__ == 'lstm' and
                        isinstance(arguments.args[4], bool)):
                    # Special case - we need this information for the lstm
                    # operation
                    arguments.special['batch_sizes'] = args[1].tolist()

                print(f"Profiling opeartion : {func.__name__}")
                forward, backward = self._profiler.measure_operation(
                    func,
                    args,
                    kwargs,
                )
                fwd_ms = forward._run_time_ms if forward is not None else None
                kernels = []
                if forward is not None:
                    kernels.extend([k.name for k in forward._kernels])
                bwd_ms = backward._run_time_ms if backward is not None else None
                if backward is not None:
                    kernels.extend([k.name for k in backward._kernels])
                print(f"Operation profiled.. exec hooked fn : {func.__name__}, fwd={fwd_ms}, bwd={bwd_ms}, kernels={kernels}")
                self._operations.append(MeasuredOperation(
                    name=func.__name__,
                    arguments=arguments,
                    forward=forward,
                    backward=backward,
                    device=self._device,
                ))

                # Actually run the hooked function
                result = func(*args, **kwargs)
                print(f"Finished hooked fn exec")
                return result
            finally:
                self._processing_hook = False
                print(f"Failed hooked fn exec")

        return hook
