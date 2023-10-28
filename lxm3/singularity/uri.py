# Port of https://github.com/sylabs/singularity/blob/main/internal/pkg/util/uri/uri.go
from typing import Optional

# Library is the keyword for a library ref
Library = "library"
# Shub is the keyword for a shub ref
Shub = "shub"
# HTTP is the keyword for http ref
HTTP = "http"
# HTTPS is the keyword for https ref
HTTPS = "https"
# Oras is the keyword for an oras ref
Oras = "oras"

_VALID_URIS: dict[str, bool] = {
    "library": True,
    "shub": True,
    "docker": True,
    "docker-archive": True,
    "docker-daemon": True,
    "oci": True,
    "oci-archive": True,
    "http": True,
    "https": True,
    "oras": True,
}


# is_valid returns whether or not the given source is valid
def is_valid(source: str) -> tuple[bool, Optional[str]]:
    u = source.split(":", 2)

    if len(u) != 2:
        return False, f"invalid uri {source}"

    if u[0] in _VALID_URIS:
        return True, ""

    return False, f"invalid uri {source}"


def filename(uri: str, suffix: str) -> str:
    """
    filename turns a transport:ref URI into a filename containing the top-level
    identifier of the image. For example, docker://sylabsio/lolcow:latest returns
    lolcow_latest.<suffix>

    Returns "" when not in transport:ref format
    """
    transport, ref = split(uri)
    if transport == "":
        return ""

    ref = ref.lstrip("/")  # Trim leading "/" characters
    refSplit = ref.split("/")  # Split ref into parts

    if transport == HTTP or transport == HTTPS:
        imageName = refSplit[len(refSplit) - 1]
        return imageName

    # Default tag is latest
    tags = ["latest"]
    container = refSplit[len(refSplit) - 1]

    if ":" in container:
        imageParts = container.split(":")
        container = imageParts[0]
        tags = [imageParts[1]]
        if "," in tags[0]:
            tags = tags[0].split(",")

    return "{}_{}.{}".format(container, tags[0], suffix)


def split(uri: str) -> tuple[str, str]:
    """
    Split splits a URI into it's components which can be used directly through containers/image
    This can be tricky if there is no type but a file name contains a colon.

    Examples:
        docker://ubuntu -> docker, //ubuntu
        docker://ubuntu:18.04 -> docker, //ubuntu:18.04
        oci-archive:path/to/archive -> oci-archive, path/to/archive
        ubuntu -> "", ubuntu
        ubuntu:18.04.img -> "", ubuntu:18.04.img

    Returns:
        transport: str, ref: str
    """
    uriSplit = uri.split(":", 1)
    if len(uriSplit) == 1:
        # no colon
        return "", uri

    if uriSplit[1].startswith("//"):
        # the format was ://, so try it whether or not valid URI
        return uriSplit[0], uriSplit[1]

    ok, err = is_valid(uri)
    if ok and not err:
        # also accept recognized URIs
        return uriSplit[0], uriSplit[1]

    return "", uri
