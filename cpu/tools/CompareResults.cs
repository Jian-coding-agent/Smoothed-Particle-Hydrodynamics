// ============================================================================
//  CompareResults -- compare two sets of SPH result<N>.dat files point by point
//
//  Usage:  CompareResults.exe <dirA> <dirB> [--tol 1e-12] [--csv out.csv]
//
//  Output:
//    1) one line per result<N>.dat: index lines nDiff maxAbs maxRel overTol
//    2) a per-column summary at the end: how many values differ in each column,
//       with the maximum absolute and relative difference
//       (columns: x0 x1 x2 p vx0 vx1 vx2 rho mass itype X00 X01 X02 rho0)
//
//  Build:  csc /nologo /optimize+ /out:CompareResults.exe CompareResults.cs
// ============================================================================
using System;
using System.Globalization;
using System.IO;
using System.Text;

static class CompareResults
{
    static readonly string[] ColName =
    {
        "x0", "x1", "x2", "p", "vx0", "vx1", "vx2",
        "rho", "mass", "itype", "X00", "X01", "X02", "rho0"
    };

    static int Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("usage: CompareResults <dirA> <dirB> [--tol 1e-12] [--csv out.csv]");
            return 2;
        }

        string dirA = args[0];
        string dirB = args[1];
        double tol = 1e-12;
        string csv = null;

        for (int i = 2; i < args.Length; i++)
        {
            if (args[i] == "--tol" && i + 1 < args.Length)
                tol = double.Parse(args[i + 1], CultureInfo.InvariantCulture);
            else if (args[i] == "--csv" && i + 1 < args.Length)
                csv = args[i + 1];
        }

        int ncol = ColName.Length;
        long[] colDiff = new long[ncol];
        double[] colMaxAbs = new double[ncol];
        double[] colMaxRel = new double[ncol];

        Console.WriteLine("A = " + Path.GetFullPath(dirA));
        Console.WriteLine("B = " + Path.GetFullPath(dirB));
        Console.WriteLine("tolerance (relative) = " + tol.ToString("G3", CultureInfo.InvariantCulture));
        Console.WriteLine();
        Console.WriteLine("index\tlines\tnDiff\tmaxAbs\tmaxRel\toverTol");

        var sb = new StringBuilder();
        sb.AppendLine("index,lines,nDiff,maxAbs,maxRel,overTol,firstDiffLine,firstDiffCol");

        for (int n = 0; ; n++)
        {
            string fa = Path.Combine(dirA, "result" + n + ".dat");
            string fb = Path.Combine(dirB, "result" + n + ".dat");
            bool ea = File.Exists(fa), eb = File.Exists(fb);
            if (!ea && !eb) break;

            if (!ea || !eb)
            {
                Console.WriteLine("{0}\tMISSING\t{1}", n, ea ? "only in A" : "only in B");
                sb.AppendLine(n + ",MISSING,," + (ea ? "only in A" : "only in B") + ",,,");
                continue;
            }

            int lines = 0, firstLine = -1, firstCol = -1;
            long nDiff = 0, overTol = 0;
            double maxAbs = 0.0, maxRel = 0.0;

            using (var ra = new StreamReader(fa))
            using (var rb = new StreamReader(fb))
            {
                string la, lb;
                while ((la = ra.ReadLine()) != null)
                {
                    lb = rb.ReadLine();
                    lines++;
                    if (lb == null) break;
                    if (la == lb) continue;                    // identical line: skip

                    string[] va = la.Split('\t');
                    string[] vb = lb.Split('\t');
                    int m = Math.Min(va.Length, vb.Length);
                    for (int c = 0; c < m; c++)
                    {
                        if (va[c] == vb[c]) continue;          // identical value: skip
                        double x = double.Parse(va[c], CultureInfo.InvariantCulture);
                        double y = double.Parse(vb[c], CultureInfo.InvariantCulture);
                        double d = Math.Abs(x - y);
                        double den = Math.Max(Math.Abs(x), Math.Abs(y));
                        double rel = den > 0.0 ? d / den : 0.0;

                        if (c < ncol)
                        {
                            colDiff[c]++;
                            if (d > colMaxAbs[c]) colMaxAbs[c] = d;
                            if (rel > colMaxRel[c]) colMaxRel[c] = rel;
                        }

                        nDiff++;
                        if (d > maxAbs) maxAbs = d;
                        if (rel > maxRel) maxRel = rel;
                        if (rel > tol) overTol++;
                        if (firstLine < 0) { firstLine = lines; firstCol = c + 1; }
                    }
                }
            }

            Console.WriteLine("{0}\t{1}\t{2}\t{3:G4}\t{4:G4}\t{5}",
                n, lines, nDiff, maxAbs, maxRel, overTol);
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "{0},{1},{2},{3:G4},{4:G4},{5},{6},{7}",
                n, lines, nDiff, maxAbs, maxRel, overTol, firstLine, firstCol));
        }

        Console.WriteLine();
        Console.WriteLine("--- per-column summary (all result files combined) ---");
        Console.WriteLine("col\tname\tdiffCount\tmaxAbs\tmaxRel");
        for (int c = 0; c < ncol; c++)
        {
            Console.WriteLine("{0}\t{1}\t{2}\t{3:G4}\t{4:G4}",
                c + 1, ColName[c], colDiff[c], colMaxAbs[c], colMaxRel[c]);
        }

        if (csv != null)
        {
            File.WriteAllText(csv, sb.ToString(), new UTF8Encoding(false));
            Console.WriteLine();
            Console.WriteLine("csv written: " + Path.GetFullPath(csv));
        }
        return 0;
    }
}
