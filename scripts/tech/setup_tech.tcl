# Set technology library

# Get list all files in current folder and check that there is a file called ${TECHNOLOGY}.tcl
if {[ file exists "scripts/tech/${TECHNOLOGY}.tcl" ]} {
    source scripts/tech/${TECHNOLOGY}.tcl
} else {
    echo "Technology ${TECHNOLOGY} not found! Please create a file called ${TECHNOLOGY}.tcl in the folder scripts/tech/ with the technology library information."
    exit 1
}
