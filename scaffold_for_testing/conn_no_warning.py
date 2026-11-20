import numpy as np
from tvb.basic.readers import H5Reader, ZipReader, try_get_absolute_path
from tvb.simulator.lab import connectivity

import logging


class ConnNoWarnings(connectivity.Connectivity):
    """
    An alteration of the connectivity from TVB which produces less errors
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_file(cls, source_file="connectivity_76.zip"):

        result = connectivity.Connectivity()
        source_full_path = try_get_absolute_path("tvb_data.connectivity", source_file)

        if source_file.endswith(".h5"):
            reader = H5Reader(source_full_path)
            reader.logger.setLevel(logging.ERROR)

            result.weights = reader.read_field("weights")
            result.centres = reader.read_field("centres")
            result.region_labels = reader.read_field("region_labels")
            result.orientations = reader.read_optional_field("orientations")
            result.cortical = reader.read_optional_field("cortical")
            result.hemispheres = reader.read_field("hemispheres")
            result.areas = reader.read_optional_field("areas")
            result.tract_lengths = reader.read_field("tract_lengths")

        else:
            reader = ZipReader(source_full_path)
            reader.logger.setLevel(logging.ERROR)
            result = cls._read(reader)

        result.weights[-1] = np.random.choice([0, 2], size=result.weights.shape[0])
        return result
