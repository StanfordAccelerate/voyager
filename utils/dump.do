# dump.do - VCS FSDB dump and runtime control script

# FSDB dump control
if {[info exists ::env(SIM_DUMP_FSDB)]} {
    echo "SIM_DUMP_FSDB is set. Enabling FSDB dump..."

    fsdbDumpfile "dump.fsdb"
    fsdbDumpvars 0 "sc_main"
    fsdbDumpon
} else {
    echo "SIM_DUMP_FSDB is not set. Skipping FSDB dump."
}

# Simulation runtime control
if {[info exists ::env(SIM_RUNTIME)]} {
    set runtime $::env(SIM_RUNTIME)
    run $runtime
    quit
} else {
    echo "SIM_RUNTIME not set. Running until simulation finishes..."
    run
}
