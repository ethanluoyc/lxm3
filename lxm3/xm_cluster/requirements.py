# type: ignore
import enum
import itertools
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import immutabledict

from lxm3._vendor.xmanager.xm import pattern_matching as pm


class _CaseInsensetiveResourceTypeMeta(enum.EnumMeta):
    """Metaclass which allows case-insensetive enum lookup.

    Enum keys are upper case, but we allow other cases for the input. For
    example existing flags and JobRequirements use lower case for resource names.
    """

    def __getitem__(cls, resource_name: str) -> "ResourceType":
        try:
            return super().__getitem__(resource_name.upper())
        except KeyError:
            raise KeyError(
                f"Unknown {cls.__name__} {resource_name!r}"
            )  # pylint: disable=raise-missing-from


class ResourceType(enum.Enum, metaclass=_CaseInsensetiveResourceTypeMeta):
    """Type of a countable resource (e.g., CPU, memory, accelerators etc).

    We use a schema in which every particular accelerator has its own type. This
    way all countable resources required for a job could be represented by a
    simple dictionary.
    """

    CPU = 100002
    RAM = 39

    # GPUs
    GPU = 100006

    # TODO: do we need V2_DONUT and V3_DONUT?

    def __str__(self):
        return self._name_


def _enum_subset(
    class_name: str, values: Iterable[ResourceType]
) -> type:  # pylint: disable=g-bare-generic
    """Returns an enum subset class.

    The class is syntactically equivalent to an enum with the given resource
    types. But the concrete constants are the same as in the ResourceType enum,
    making all equivalence comparisons work correctly. Additionally operator `in`
    is supported for checking if a resource belongs to the subset.

    Args:
      class_name: Class name of the subset enum.
      values: A list of resources that belong to the subset.
    """
    values = set(values)

    class EnumSubsetMetaclass(type):  # pylint: disable=g-bare-generic
        """Metaclass which implements enum subset operations."""

        def __new__(
            cls,
            name: str,
            bases: Tuple[type],  # pylint: disable=g-bare-generic
            dct: Dict[str, Any],
        ) -> type:  # pylint: disable=g-bare-generic
            # Add constants to the class dict.
            for name, member in ResourceType.__members__.items():
                if member in values:
                    dct[name] = member

            return super().__new__(cls, class_name, bases, dct)

        def __getitem__(cls, item: str) -> ResourceType:
            result = ResourceType[item]
            if result not in cls:  # pylint: disable=unsupported-membership-test
                raise AttributeError(
                    f"type object '{cls.__name__}' has no attribute '{item}'"
                )
            return result

        def __iter__(cls) -> Iterator[ResourceType]:
            return iter(values)

        def contains(cls, value: ResourceType) -> bool:
            return value in values

    class EnumSubset(metaclass=EnumSubsetMetaclass):
        def __new__(cls, value: int) -> ResourceType:
            resource = ResourceType(value)
            if resource not in cls:
                raise ValueError(f"{value} is not a valid {cls.__name__}")
            return resource

    return EnumSubset


_AcceleratorType = _enum_subset(
    "_AcceleratorType",
    [
        ResourceType.GPU,
    ],
)


class ResourceDict(MutableMapping):
    """Internal class to represent amount of countable resources.

    A mapping from ResourceType to amount of the resource combined with
    convenience methods. This class only tracks amounts of the resources, but not
    their topologies, locations or constraints.

    This class is rather generic and is designed be used internally as job
    requirements as well as in the executors. API users should not use it
    explicitly.

    Usage:
      # Construct (implicitly) from user code using JobRequirements:
      requirements = JobRequirements(cpu=0.5 * xm.vCPU, memory=2 * xm.GiB, v100=8)
      resources = requirements.task_requirements
      # Resources are available by their canonical names.
      assert(resources[ResourceType.V100], 8)
      # Print user-friendly representation:
      print(f'The task needs {resources}')
    """

    def __init__(self) -> None:
        self.__dict: Dict[ResourceType, float] = {}

    def __setitem__(self, key: ResourceType, value: float) -> None:
        self.__dict.__setitem__(key, value)

    def __getitem__(self, key: ResourceType) -> float:
        return self.__dict.__getitem__(key)

    def __delitem__(self, key: ResourceType) -> None:
        self.__dict.__delitem__(key)

    def __iter__(self):
        return self.__dict.__iter__()

    def __len__(self) -> int:
        return self.__dict.__len__()

    def __str__(self) -> str:
        """Returns user-readable text representation.

        Such as "V100: 8, CPU: 1.2, MEMORY: 5.4GiB".
        """
        # TODO: We do not aggregate memory yet, update this method to be more
        # user-friendly.
        return ", ".join(sorted([f"{key}: {value}" for (key, value) in self.items()]))

    def __add__(self: "ResourceDict", rhs: "ResourceDict") -> "ResourceDict":
        """Returns a sum of two ResourceDicts."""
        result = ResourceDict()
        for key in [*self.keys(), *rhs.keys()]:
            result[key] = self.get(key, 0) + rhs.get(key, 0)
        return result

    def __mul__(self: "ResourceDict", rhs: float) -> "ResourceDict":
        """Returns the multiplication of a ResourceDict with a scalar."""
        result = ResourceDict()
        for key, value in self.items():
            result[key] = value * rhs
        return result

    def __rmul__(self: "ResourceDict", rhs: float) -> "ResourceDict":
        """Returns the multiplication of a ResourceDict with a scalar."""
        return self * rhs


ResourceQuantity = Union[int, str]


class JobRequirements:
    # pyformat: disable
    """Describes the resource requirements of a Job.

    Attributes:
      task_requirements: Amount of resources needed for a single task within a
        job.
      accelerator: The accelerator the jobs uses, if there is one. Jobs using
        multiple accelerators are not supported because different kinds of
        accelerators are usually not installed on the same host.
      topology: Accelerator topology, if an accelerator is used.
      location: Place where the job should run. For example a cluster name or a
        Borg cell.
      service_tier: A service tier at which the job should run.
      replicas: Number of identical tasks to run within a job
    """
    # pyformat:enable

    task_requirements: ResourceDict
    accelerator: Optional[ResourceType]

    location: Optional[str]

    def __init__(
        self,
        resources: Mapping[
            Union[ResourceType, str], ResourceQuantity
        ] = immutabledict.immutabledict(),
        *,
        location: Optional[str] = None,
        **kw_resources: ResourceQuantity,
    ) -> None:
        # pyformat: disable
        """Define a set of resources.

        Args:
          resources: resource amounts as a dictionary, for example
            {xm.ResourceType.V100: 2}.
          location: Place where the job should run. For example a cluster name or a
            Borg cell.
          **kw_resources: resource amounts as a kwargs, for example `v100=2` or
            `ram=1 * xm.GiB`. See xm.ResourceType enum for the list of supported
            types and aliases.

        Raises:
          ValueError:
            If several accelerator resources are supplied (i.e. GPU and TPU).
            If the same resource is passed in a `resources` dictionary and as
              a command line argument.
            If topology is supplied for a non accelerator resource.
        """
        # pyformat: enable
        self.location = location

        self.task_requirements = ResourceDict()
        self.accelerator = None
        self.topology = None

        for resource_name, value in itertools.chain(
            resources.items(), kw_resources.items()
        ):
            resource = pm.match(
                pm.Case([str], lambda r: ResourceType[r]),
                pm.Case([ResourceType], lambda r: r),
            )(resource_name)

            if resource in _AcceleratorType:
                if self.accelerator is not None:
                    raise ValueError("Accelerator already set.")
                self.accelerator = resource

            if resource in self.task_requirements:
                raise ValueError(f"{resource} has been specified twice.")
            self.task_requirements[resource] = value

    def __repr__(self) -> str:
        """Returns string representation of the requirements."""
        args = []

        if self.location:
            args.append(f"location={self.location!r}")

        return f'xm.JobRequirements({", ".join(args)})'
