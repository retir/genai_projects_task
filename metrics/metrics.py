from utils.class_registry import ClassRegistry


metrics_registry = ClassRegistry()


@metrics_registry.add_to_registry(name="fid")
class FID:
    def __call__(self, orig_pth, synt_pth):
        # TO DO
        # fid = ...
        # return fid
        raise NotImplementedError()