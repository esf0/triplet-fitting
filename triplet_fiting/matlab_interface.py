"""
Copyright (c) 2024 Egor Sedov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

1. Attribution: You must give appropriate credit, provide a link to the license,
   and indicate if changes were made. You may do so in any reasonable manner, but
   not in any way that suggests the licensor endorses you or your use.

2. No Cloning without Citation: You are not allowed to clone this repository or
   any substantial part of it without citing the author. Citation must include
   the author's name (Egor Sedov) and contact email (egor.sedoff+git@gmail.com).

3. The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import matlab.engine

def start_matlab_engine() -> matlab.engine.MatlabEngine:
    return matlab.engine.start_matlab()
