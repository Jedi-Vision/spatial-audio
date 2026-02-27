# Socket Protocol v1

Protocol v1 is binary and little-endian. Frame envelope:

1. `int32 frame_number`
2. `double timestamp_ms`
3. `char '^'` list start marker
4. `int32 object_count`
5. repeated objects prefixed by `char '|'`
6. optional trailing `char '^'`

## Object Layouts

### 2D frame object (`frame_2d_v1`)
- `int32 id`
- `int32 label`
- `double x_2d`
- `double y_2d`
- `double depth`

### 3D frame object (`frame_3d_v1`)
- `int32 id`
- `int32 label`
- `double x`
- `double y`
- `double z`

## Parse Fail Conditions
- null pointer
- payload shorter than header minimum
- missing `'^'` or `'|'` markers
- negative object count
- object count over parser safety limits
- truncated object record fields
- trailing bytes that do not match optional terminal `'^'`

## ACK Convention
- `'0'`: payload parsed and accepted
- `'1'`: payload parse failure
