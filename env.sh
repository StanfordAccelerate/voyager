eval `modulecmd bash load base catapult/2024.2 vcs/T-2022.06-SP2 verdi/T-2022.06-SP2`
export VG_GNU_PACKAGE=/cad/synopsys/vcs_gnu_package/S-2021.09/gnu_9/linux
source /cad/synopsys/vcs_gnu_package/S-2021.09/gnu_9/linux/source_me.sh
export LD_LIBRARY_PATH=/cad/mentor/2024.2/Mgc_home/lib:/cad/mentor/2024.2/Mgc_home/shared/lib:$LD_LIBRARY_PATH
# TODO: both of these can be moved to the makefile and run_regression.sh
export PROJECT_ROOT=`pwd`
export CODEGEN_DIR=test/compiler
