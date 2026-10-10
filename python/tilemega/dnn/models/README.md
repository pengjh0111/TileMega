NAFNet sources are copied byte for byte from the commit and paths in
`upstream.json`. The architecture, including its autograd LayerNorm2d, is
unchanged. `NAFNet_LICENSE` preserves the upstream MIT license and notices.

`_upstream` contains the architecture's original import dependencies. Its
small `basicsr.utils` initializer exposes the logger without importing the
upstream training framework. It does not implement or replace model operators.

The SIDD width32 configuration is width=32, middle blocks=12, encoder
blocks=[2,2,4,8], decoder blocks=[2,2,2,2]. The official checkpoint must load
with `strict=True`; structure-only exports cannot enter accuracy or timing
tables.
