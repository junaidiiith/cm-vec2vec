from serialization import (
    serialize_bpmn_models,
    serialize_ontouml_models,
    serialize_archimate_models,
)

serialize_bpmn_models('datasets/bpmn_chunks')
serialize_ontouml_models('datasets/ontouml-models', 'ontouml-models.csv', 'ontouml_models_serialized.csv')
serialize_archimate_models('datasets/eamodelset', 'eamodelset_serialized.csv')