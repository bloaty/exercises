## What

This repo contains Java solutions to various programming exercises.
Bring your own `java` and `javac`. Java 8+ should suffice.

## How to Run

I don't expect to need much apparatus beyond the Java standard
library, and maybe a very small handful of helpful dependencies
like Guava. Therefore, it didn't make sense to set up elaborate
config to compile and run the programs.

Compile everything using your own `javac` into `out/` like so:
```shell
  javac -cp libs -d out $(find src -iname "*.java")
```
Run individual programs like so:
```shell
  java [JVM_OPTS...] -cp 'out:libs' [CLASS] [ARGS...]
```
Above, `CLASS` is the full name of the program class, which is
the class file's relative path from `out/`, minus the `.class`
extension. For example: `compprog3e.PairingProblem`.

`ARGS` is a space-separated list of program arguments. This is
typically the path to an input file. Comments in the source
specify what is expected.

Supply `JVM_OPTS` to the JVM as you see fit: `-Xms###m` and
`-Xmx###m` for min/max heap size, `-XX:+HeapDumpOnOutOfMemoryError`
with `-XX:HeapDumpPath=###` if you want to do heap analysis, etc.
