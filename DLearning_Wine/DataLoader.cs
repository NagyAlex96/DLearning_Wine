using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DLearning_Wine
{
    public class DataLoader
    {
        private static int inputDim = 11;      // Bemeneti jellemzők száma

        // Adatok betöltése és előkészítése az edzéshez
        public List<Tuple<float[], float[]>> LoadWineData(string path, int numOutputClasses)
        {
            var data = new List<Tuple<float[], float[]>>();
            foreach (var line in File.ReadAllLines(path).Skip(1))
            {
                var values = line.Split(';').Select(x => float.Parse(x, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
                var features = values.Take(inputDim).ToArray();
                var label = new float[numOutputClasses];
                label[(int)values.Last()] = 1.0f;
                data.Add(new Tuple<float[], float[]>(features, label));
            }
            return data;
        }
    }
}
