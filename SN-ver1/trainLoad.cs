using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SN_ver1
{
     class trainLoad
    {
        [LoadColumn(0)]
      public DateTime data;

        [LoadColumn(1)]
        public float open;

        [LoadColumn(2)]
        public float high;

        [LoadColumn(3)]
        public float low;

        [LoadColumn(4)]
        public float close;

        [LoadColumn(5)]
        public float adjClose;

        [LoadColumn(6)]
        public double Volume;


    }
}
