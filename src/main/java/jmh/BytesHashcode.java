/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package jmh;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
public class BytesHashcode {
    static final VectorSpecies<Integer> INT_256_SPECIES = IntVector.SPECIES_256;

    static final VectorSpecies<Byte> BYTE_64_SPECIES = ByteVector.SPECIES_64;
    static final VectorSpecies<Byte> BYTE_128_SPECIES = ByteVector.SPECIES_128;
    static final VectorSpecies<Byte> BYTE_256_SPECIES = ByteVector.SPECIES_256;

    static final int COEFF_31_TO_8;
    static final int COEFF_31_TO_16;
    static final int COEFF_31_TO_32;

    static final IntVector H_COEFF_31_TO_8;
    static final IntVector H_COEFF_31_TO_16;
    static final IntVector H_COEFF_31_TO_32;

    static final IntVector H_COEFF_8;
    static final IntVector H_COEFF_16;
    static final IntVector H_COEFF_24;
    static final IntVector H_COEFF_32;


    static {
        int[] x = new int[INT_256_SPECIES.length() * 4];
        x[x.length - 1] = 1;
        for (int i = 1; i < x.length; i++) {
            x[x.length - 1 - i] = x[x.length - 1 - i + 1] * 31;
        }

        COEFF_31_TO_8 = x[24] * 31;
        COEFF_31_TO_16 = x[16] * 31;
        COEFF_31_TO_32 = x[0] * 31;

        H_COEFF_31_TO_8 = IntVector.broadcast(INT_256_SPECIES, COEFF_31_TO_8);
        H_COEFF_31_TO_16 = IntVector.broadcast(INT_256_SPECIES, COEFF_31_TO_16);
        H_COEFF_31_TO_32 = IntVector.broadcast(INT_256_SPECIES, COEFF_31_TO_32);

        H_COEFF_8 = IntVector.fromArray(INT_256_SPECIES, x, 24);
        H_COEFF_16 = IntVector.fromArray(INT_256_SPECIES, x, 16);
        H_COEFF_24 = IntVector.fromArray(INT_256_SPECIES, x, 8);
        H_COEFF_32 = IntVector.fromArray(INT_256_SPECIES, x, 0);
    }

    @Param("1024")
    int size;

    byte[] a;

    @Setup
    public void init() {
        a = new byte[size];
        ThreadLocalRandom.current().nextBytes(a);
    }


    @Benchmark
    public int scalar() {
        return Arrays.hashCode(a);
    }

    /*
        Hashcode calculation can be represented a polynomial

        h = 31^l
          + 31^(l - 1) * a[0]
          + 31^(l - 2) * a[1]
          + ...
          + 31^2 * a[l - 3]
          + 31 * a[l - 2]
          + a[l - 1]

     */


    @Benchmark
    public int scalarUnrolled() {
        if (a == null)
            return 0;

        int h = 1;
        int i = 0;
        for (; i < (a.length & ~(8 - 1)); i += 8) {
            h = h * 31 * 31 * 31 * 31 * 31 * 31 * 31 * 31 +
                    a[i + 0] * 31 * 31 * 31 * 31 * 31 * 31 * 31 +
                    a[i + 1] * 31 * 31 * 31 * 31 * 31 * 31 +
                    a[i + 2] * 31 * 31 * 31 * 31 * 31 +
                    a[i + 3] * 31 * 31 * 31 * 31 +
                    a[i + 4] * 31 * 31 * 31 +
                    a[i + 5] * 31 * 31 +
                    a[i + 6] * 31 +
                    a[i + 7];
        }

        for (; i < a.length; i++) {
            h = 31 * h + a[i];
        }
        return h;
    }

    @Benchmark
    public int vector64ReduceInLoop() {
        int h = 1;
        int i = 0;
        for (; i < BYTE_64_SPECIES.loopBound(a.length); i += BYTE_64_SPECIES.length()) {
            // load 8 bytes, into a 64-bit vector
            ByteVector b = ByteVector.fromArray(BYTE_64_SPECIES, a, i);
            // convert 8 bytes into 8 ints, into a 256-bit vector
            IntVector x = (IntVector) b.castShape(INT_256_SPECIES, 0);
            h = h * COEFF_31_TO_8 + x.mul(H_COEFF_8).reduceLanes(VectorOperators.ADD);
        }

        for (; i < a.length; i++) {
            h = 31 * h + a[i];
        }
        return h;
    }

    @Benchmark
    public int vector64() {
        IntVector h = IntVector.fromArray(INT_256_SPECIES, new int[]{1, 0, 0, 0, 0, 0, 0, 0}, 0);
        int i = 0;
        for (; i < BYTE_64_SPECIES.loopBound(a.length); i += BYTE_64_SPECIES.length()) {
            ByteVector b = ByteVector.fromArray(BYTE_64_SPECIES, a, i);
            IntVector x = (IntVector) b.castShape(INT_256_SPECIES, 0);
            h = h.mul(H_COEFF_31_TO_8).add(x.mul(H_COEFF_8));
        }

        int sh = h.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            sh = 31 * sh + a[i];
        }
        return sh;
    }

    @Benchmark
    public int vector64Unrolledx2() {
        IntVector h1 = IntVector.fromArray(INT_256_SPECIES, new int[]{1, 0, 0, 0, 0, 0, 0, 0}, 0);
        IntVector h2 = IntVector.zero(INT_256_SPECIES);
        int i = 0;
        for (; i < (a.length & ~(BYTE_128_SPECIES.length() - 1)); i += BYTE_128_SPECIES.length()) {
            ByteVector b = ByteVector.fromArray(BYTE_64_SPECIES, a, i);
            IntVector x = (IntVector) b.castShape(INT_256_SPECIES, 0);
            h1 = h1.mul(H_COEFF_31_TO_16).add(x.mul(H_COEFF_16));

            b = ByteVector.fromArray(BYTE_64_SPECIES, a, i + BYTE_64_SPECIES.length());
            x = (IntVector) b.castShape(INT_256_SPECIES, 0);
            h2 = h2.mul(H_COEFF_31_TO_16).add(x.mul(H_COEFF_8));
        }

        int sh = h1.reduceLanes(VectorOperators.ADD) + h2.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            sh = 31 * sh + a[i];
        }
        return sh;
    }


    @Benchmark
    public int vector64Unrolledx4() {
        IntVector h1 = IntVector.fromArray(INT_256_SPECIES, new int[]{1, 0, 0, 0, 0, 0, 0, 0}, 0);
        IntVector h2 = IntVector.zero(INT_256_SPECIES);
        IntVector h3 = IntVector.zero(INT_256_SPECIES);
        IntVector h4 = IntVector.zero(INT_256_SPECIES);
        int i = 0;
        for (; i < (a.length & ~(BYTE_256_SPECIES.length() - 1)); i += BYTE_256_SPECIES.length()) {
            ByteVector b = ByteVector.fromArray(BYTE_64_SPECIES, a, i);
            IntVector x = (IntVector) b.castShape(INT_256_SPECIES, 0);
            h1 = h1.mul(H_COEFF_31_TO_32).add(x.mul(H_COEFF_32));

            b = ByteVector.fromArray(BYTE_64_SPECIES, a, i + BYTE_64_SPECIES.length());
            x = (IntVector) b.castShape(INT_256_SPECIES, 0);
            h2 = h2.mul(H_COEFF_31_TO_32).add(x.mul(H_COEFF_24));

            b = ByteVector.fromArray(BYTE_64_SPECIES, a, i + BYTE_64_SPECIES.length() * 2);
            x = (IntVector) b.castShape(INT_256_SPECIES, 0);
            h3 = h3.mul(H_COEFF_31_TO_32).add(x.mul(H_COEFF_16));

            b = ByteVector.fromArray(BYTE_64_SPECIES, a, i + BYTE_64_SPECIES.length() * 3);
            x = (IntVector) b.castShape(INT_256_SPECIES, 0);
            h4 = h4.mul(H_COEFF_31_TO_32).add(x.mul(H_COEFF_8));
        }

        int sh = h1.reduceLanes(VectorOperators.ADD) +
                h2.reduceLanes(VectorOperators.ADD) +
                h3.reduceLanes(VectorOperators.ADD) +
                h4.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            sh = 31 * sh + a[i];
        }
        return sh;
    }
}
/*

$ git rev-parse --short HEAD
dfacda488bf

java -XX:-TieredCompilation -jar target/benchmarks.jar BytesHashcode

# JMH version: 1.31
# VM version: JDK 17-internal, OpenJDK 64-Bit Server VM, 17-internal+0-adhoc.sandoz.jdk17
# VM invoker: /Users/sandoz/Projects/jdk/jdk17/build/macosx-x86_64-server-release/images/jdk/bin/java
# VM options: --add-modules=jdk.incubator.vector -XX:-TieredCompilation

Benchmark                           (size)   Mode  Cnt     Score     Error   Units
BytesHashcode.scalar                  1024  thrpt    5  1280.707 ±   3.982  ops/ms
BytesHashcode.scalarUnrolled          1024  thrpt    5   573.338 ±  22.649  ops/ms
BytesHashcode.vector64                1024  thrpt    5  2636.888 ±   8.721  ops/ms
BytesHashcode.vector64ReduceInLoop    1024  thrpt    5  3855.475 ±  28.767  ops/ms
BytesHashcode.vector64Unrolledx2      1024  thrpt    5  5134.652 ±  42.714  ops/ms
BytesHashcode.vector64Unrolledx4      1024  thrpt    5  8733.027 ± 255.440  ops/ms


java -XX:-TieredCompilation -XX:LoopUnrollLimit=0 -jar target/benchmarks.jar -prof dtraceasm BytesHashcode.vector64$

Hot loop:

  5.96%  ↗│  0x000000011a1dff50:   vmovq  0x10(%r9,%r8,1),%xmm1     // b = ByteVector.fromArray
         ││  0x000000011a1dff57:   movabs $0x61d418178,%rdi         // Address of vector constant H_COEFF_31_TO_8
  2.56%  ││  0x000000011a1dff61:   vpmulld 0x10(%rdi),%ymm0,%ymm0   // h = h.mul(H_COEFF_31_TO_8)
 69.97%  ││  0x000000011a1dff67:   vpmovsxbd %xmm1,%ymm1            // x = b.castShape(INT_256_SPECIES, 0)
         ││  0x000000011a1dff6c:   movabs $0x61d4181a8,%rdi         // Address of vector constant H_COEFF_8
  0.02%  ││  0x000000011a1dff76:   vpmulld 0x10(%rdi),%ymm1,%ymm1   // x = x.mul(H_COEFF_8)
  8.58%  ││  0x000000011a1dff7c:   vpaddd %ymm1,%ymm0,%ymm0         // y = y + x
  8.50%  ││  0x000000011a1dff80:   add    $0x8,%r8d                 //
         ││  0x000000011a1dff84:   cmp    %r11d,%r8d                // next 8 bytes
         ╰│  0x000000011a1dff87:   jl     0x000000011a1dff50        //

 */