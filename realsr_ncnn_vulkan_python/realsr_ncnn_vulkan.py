import sys
from math import floor
from pathlib import Path

import numpy as np

if __package__:
    import importlib

    raw = importlib.import_module(f"{__package__}.realsr_ncnn_vulkan_wrapper")
else:
    import realsr_ncnn_vulkan_wrapper as raw


class RealSR:
    def __init__(
            self,
            gpuid=0,
            model="models-DF2K",
            tta_mode=False,
            scale: float = 2,
            tilesize=0,
            param_path="test.param",
            bin_path="test.bin",
    ):
        """
        RealSR class which can do image super resolution.

        :param gpuid: the id of the gpu device to use.
        :param model: the name or the path to the model
        :param tta_mode: whether to enable tta mode or not
        :param scale: scale ratio. value: float. default: 2
        :param tilesize: tile size. 0 for automatically setting the size. default: 0
        """
        self._raw_realsr = raw.RealSRWrapped(gpuid, tta_mode)
        self.model = model
        self.gpuid = gpuid
        self.scale = scale  # the real scale ratio
        self.set_params(scale, tilesize)
        self.load(param_path, bin_path)

    def set_params(self, scale=4., tilesize=0):
        """
        set parameters for realsr object

        :param scale: 1/2. default: 2
        :param tilesize: default: 0
        :return: None
        """
        self._raw_realsr.scale = scale  # control the real scale ratio at each raw process function call
        self._raw_realsr.tilesize = self.get_tilesize() if tilesize <= 0 else tilesize
        self._raw_realsr.prepadding = self.get_prepadding()

    def load(self, parampath: str = "", modelpath: str = "") -> None:
        """
        Load models from given paths. Use self.model if one or all of the parameters are not given.

        :param parampath: the path to model params. usually ended with ".param"
        :param modelpath: the path to model bin. usually ended with ".bin"
        :return: None
        """
        # cant delete this, otherwise it wont work
        if not parampath or not modelpath:
            model_dir = Path(self.model)
            if not model_dir.is_absolute():
                if (
                        not model_dir.is_dir()
                ):  # try to load it from module path if not exists as directory
                    dir_path = Path(__file__).parent
                    model_dir = dir_path.joinpath("models", self.model)

            if self._raw_realsr.scale == 4:
                parampath = model_dir.joinpath("x4.param")
                modelpath = model_dir.joinpath("x4.bin")

        if Path(parampath).exists() and Path(modelpath).exists():
            parampath_str, modelpath_str = raw.StringType(), raw.StringType()
            if sys.platform in ("win32", "cygwin"):
                parampath_str.wstr = raw.new_wstr_p()
                raw.wstr_p_assign(parampath_str.wstr, str(parampath))
                modelpath_str.wstr = raw.new_wstr_p()
                raw.wstr_p_assign(modelpath_str.wstr, str(modelpath))
            else:
                parampath_str.str = raw.new_str_p()
                raw.str_p_assign(parampath_str.str, str(parampath))
                modelpath_str.str = raw.new_str_p()
                raw.str_p_assign(modelpath_str.str, str(modelpath))

            self._raw_realsr.load(parampath_str, modelpath_str)
        else:
            raise FileNotFoundError(f"{parampath} or {modelpath} not found")

    def worker(self, q, im):
        try:
            # Do stuff here with usual try/except/raise
            if self.scale > 1:
                cur_scale = 1
                self.w = im.shape[1]
                self.h = im.shape[0]
            im = self._process(im)
            q.put(im)
        except Exception as e:
            q.put(e)

    def process(self, img):
        import queue

        import multiprocess as mp

        q = mp.Queue()
        p = mp.Process(target=self.worker, args=(q, img))
        p.start()
        p.join()

        # Get results from worker
        try:
            result = q.get(block=False)
            # Check if an exception was raised by worker code
            if isinstance(result, Exception):
                raise result

            return result
        except queue.Empty:
            # Do failure handling here
            raise RuntimeError("An error occurred during NCNN processing")

    def _process(self, im):
        """
        Call RealSR.process() once for the given PIL.Image
        """
        in_bytes = bytearray(np.array(im).tobytes(order='C'))
        channels = int(len(in_bytes) / (self.w * self.h))
        out_bytes = bytearray((self._raw_realsr.scale ** 2) * len(in_bytes))

        raw_in_image = raw.Image(in_bytes, self.w, self.h, channels)
        raw_out_image = raw.Image(
            out_bytes,
            self._raw_realsr.scale * self.w,
            self._raw_realsr.scale * self.h,
            channels,
        )

        self._raw_realsr.process(raw_in_image, raw_out_image)

        out_numpy = np.frombuffer(bytes(out_bytes), dtype=np.uint8)
        out_numpy = np.reshape(
            out_numpy, (self._raw_realsr.scale * self.h, self._raw_realsr.scale * self.w, 3))
        return out_numpy

    def get_prepadding(self) -> int:
        if self.model.find("models-DF2K") or self.model.find("models-DF2K_JPEG"):
            return 10
        else:
            raise NotImplementedError(f'model "{self.model}" is not supported')

    def get_tilesize(self):
        heap_budget = raw.get_heap_budget(self.gpuid)
        if self.model.find("models-DF2K") or self.model.find("models-DF2K_JPEG"):
            if heap_budget > 1900:
                return 200
            elif heap_budget > 550:
                return 100
            elif heap_budget > 190:
                return 64
            else:
                return 32
        else:
            raise NotImplementedError(f'model "{self.model}" is not supported')
