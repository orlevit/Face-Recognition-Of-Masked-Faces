# This is constants file for run bash

################################################################ indication which train dataset is running #################################################################
declare -a name_executed_arr=('CASIA' 'CASIA_M_EYES' 'CASIA_M_HAT' 'CASIA_M_SCARF' 'CASIA_M_CORONA')

EXECUTE_CASIA=false
EXECUTE_CASIA_M_EYES=true
EXECUTE_CASIA_M_HAT=true
EXECUTE_CASIA_M_SCARF=true
EXECUTE_CASIA_M_CORONA=true

declare -a what_to_execute_arr=($EXECUTE_CASIA $EXECUTE_CASIA_M_EYES $EXECUTE_CASIA_M_HAT $EXECUTE_CASIA_M_SCARF $EXECUTE_CASIA_M_CORONA)

########################################################################### dataset name ###########################################################################

DATASET_CASIA='fame'
DATASET_CASIA_M_EYES='fame_eye_masked'
DATASET_CASIA_M_HAT='fame_hat_masked'
DATASET_CASIA_M_SCARF='fame_scarf_masked'
DATASET_CASIA_M_CORONA='fame_corona_masked'

declare -a dataset_arr=($DATASET_CASIA $DATASET_CASIA_M_EYES $DATASET_CASIA_M_HAT $DATASET_CASIA_M_SCARF $DATASET_CASIA_M_CORONA)

