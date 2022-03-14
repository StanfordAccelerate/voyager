solution design set {TreeOps<PositFP<6, 12>, 16>::treeadd} -top
solution design set {TreeOps<PositFP<6, 12>, 16>::treeadd} -combinational

go compile

solution options set ComponentLibs/SearchPath {/home/shared/catapult/memories /home/shared/catapult/stdcells} -append
solution library add tcbn40ulpbwp40_c170815tt1p1v25c_dc -- -rtlsyntool DesignCompiler -vendor TSMC -technology 40nm

go libraries

directive set -CLOCKS $clocks

go extract

solution new -state analyze

solution design set {TreeOps<PositFP<6, 12>, 16>::treemax} -top
solution design set {TreeOps<PositFP<6, 12>, 16>::treemax} -combinational

go compile

solution options set ComponentLibs/SearchPath {/home/shared/catapult/memories /home/shared/catapult/stdcells} -append
solution library add tcbn40ulpbwp40_c170815tt1p1v25c_dc -- -rtlsyntool DesignCompiler -vendor TSMC -technology 40nm

go libraries

directive set -CLOCKS $clocks

go extract