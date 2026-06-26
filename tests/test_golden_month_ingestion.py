import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline.ingestion.helios_loader import HEL1OSLoader
from pipeline.ingestion.suit_loader import SUITLoader


pytest.importorskip("astropy")


def _write_helios_fits(path: Path, detector: str) -> None:
    from astropy.io import fits

    if detector == "CdTe":
        cols = [
            fits.Column(name="TIME", array=np.array([769435200.0, 769435201.0]), format="D"),
            fits.Column(name="RATE_CdTe_5_12keV", array=np.array([1.0, 2.0]), format="D"),
            fits.Column(name="RATE_CdTe_12_25keV", array=np.array([2.0, 3.0]), format="D"),
            fits.Column(name="RATE_CdTe_25_40keV", array=np.array([4.0, 5.0]), format="D"),
            fits.Column(name="RATE_CdTe_40_80keV", array=np.array([6.0, 7.0]), format="D"),
        ]
    else:
        cols = [
            fits.Column(name="TIME", array=np.array([769435200.0, 769435201.0]), format="D"),
            fits.Column(name="RATE_CZT_20_40keV", array=np.array([8.0, 9.0]), format="D"),
            fits.Column(name="RATE_CZT_40_80keV", array=np.array([10.0, 11.0]), format="D"),
            fits.Column(name="RATE_CZT_80_120keV", array=np.array([12.0, 13.0]), format="D"),
            fits.Column(name="RATE_CZT_120_150keV", array=np.array([14.0, 15.0]), format="D"),
        ]

    primary = fits.PrimaryHDU()
    primary.header["T_START"] = "2024-05-14T00:00:00"
    primary.header["T_STOP"] = "2024-05-14T00:00:02"
    primary.header["QVAL"] = 100
    primary.header["EXPOSURE"] = 2
    primary.header["DEADCOR_CT" if detector == "CdTe" else "DEADCOR_CZ"] = 0.0
    hdul = fits.HDUList([primary, fits.BinTableHDU.from_columns(cols)])
    hdul.writeto(path)


def test_helios_load_day_merges_cdte_czt_from_zip(tmp_path: Path) -> None:
    cdte = tmp_path / "AL1_HXS91_20240514_000000_L2_CdTe_V01.fits"
    czt = tmp_path / "AL1_HXS91_20240514_000000_L2_CZT_V01.fits"
    _write_helios_fits(cdte, "CdTe")
    _write_helios_fits(czt, "CZT")

    zip_path = tmp_path / "HEL1OS_20240514.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(cdte, cdte.name)
        zf.write(czt, czt.name)
    cdte.unlink()
    czt.unlink()

    records = HEL1OSLoader(tmp_path).load_day("20240514")

    assert len(records) == 2
    assert records[0].cdte1_30_40 > 0
    assert records[0].czt1_40_60 > 0


def test_suit_scan_directory_accepts_zip_fits(tmp_path: Path) -> None:
    from astropy.io import fits

    fits_path = tmp_path / "SUIT_NB02_20240514_000000_L2_V01.fits"
    primary = fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32))
    primary.header["T_OBS"] = "2024-05-14T00:00:00"
    primary.header["FTR_NAME"] = "NB02"
    primary.header["QVAL"] = 100
    primary.writeto(fits_path)

    zip_path = tmp_path / "SUIT_20240514.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(fits_path, fits_path.name)
    fits_path.unlink()

    images = SUITLoader(tmp_path).scan_directory(date_str="20240514")

    assert len(images) == 1
    assert images[0].filter_name == "NB02"
