# Read and set relevant environment variables
set ENV_VARS {BLOCK CLOCK_PERIOD DATATYPE IC_DIMENSION OC_DIMENSION TECHNOLOGY}

foreach var $ENV_VARS {
  set $var [exec echo $::env($var)]
}