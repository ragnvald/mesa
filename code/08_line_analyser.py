#
#
#  User selects input geocode group
#  List of lines is provided (tbl_line_in)
#   1) No tbl_line_input means a line should be created based on tbl_flat.
#   2) tbl_line_input has one of more lines
#  User selects lines
#  User selects:
#     - buffered width of line in meters
#     - line segment length
#  Feedback if selected combinations will overwrite existing data.
#  Analyse overlayng geocode with line segments / polygons
#     - results in tbl_line_out_flat (we might build a stacked one at a later stage...)
#
#
#