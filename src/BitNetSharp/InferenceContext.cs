using BitNetSharp.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BitNetSharp
{
    public class InferenceContext(BitNetModel model)
    {
        public BitNetModel Model => model;
        public int[] Tokens { get; set; }
        public int CurrentToken { get; set; }
    }
}
