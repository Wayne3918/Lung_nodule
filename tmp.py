import pylidc as pl

scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1.5,
                                 pl.Scan.pixel_spacing <= 0.6)
print(scans.count())