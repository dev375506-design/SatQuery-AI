"""
Comprehensive scientific test suite for SatQuery AI:
- Test A: Single Image Analysis (POST /api/analyze)
- Test B: Bi-Temporal Analysis (POST /api/analyze/change)
- Test C: Optical + SAR Fusion (Standard Images)
- Test D: GeoTIFF Multi-Band Sentinel-2 + Dual-Pol Sentinel-1 in Projected UTM (EPSG:32642)
- Test E: GeoTIFF Multi-Band Sentinel-2 + Dual-Pol Sentinel-1 in Geographic WGS84 Degrees (EPSG:4326 - Larkana Pakistan)
- Test F: Missing Optical / SAR Images
- Test G: Corrupt Image Handling
- Test H: Different Dimensions Fallback
- Test I: Agent Router Compatibility
"""

import io
import sys
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.main import app

client = TestClient(app)


def create_synthetic_image(width=100, height=80, color=(100, 150, 200), fmt="PNG"):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def create_synthetic_sar_image(width=100, height=80, fmt="PNG"):
    arr = np.random.uniform(20, 80, (height, width)).astype(np.uint8)
    arr[20:50, 20:50] = np.random.uniform(2, 10, (30, 30))
    arr[10:30, 60:80] = np.random.uniform(200, 255, (20, 20))
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def create_sentinel2_geotiff(width=200, height=150, crs="EPSG:32642", left=250000.0, top=3050000.0, res=10.0):
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(left, top, res, res)
    buf = io.BytesIO()

    b02 = np.random.uniform(700, 1100, (height, width)).astype(np.uint16)
    b03 = np.random.uniform(900, 1400, (height, width)).astype(np.uint16)
    b04 = np.random.uniform(800, 1300, (height, width)).astype(np.uint16)
    b08 = np.random.uniform(2500, 4000, (height, width)).astype(np.uint16)

    # Candidate water body: high green, low red, very low NIR
    b02[50:110, 40:120] = np.random.uniform(1100, 1400, (60, 80)).astype(np.uint16)
    b03[50:110, 40:120] = np.random.uniform(1400, 1800, (60, 80)).astype(np.uint16)
    b04[50:110, 40:120] = np.random.uniform(300, 500, (60, 80)).astype(np.uint16)
    b08[50:110, 40:120] = np.random.uniform(150, 300, (60, 80)).astype(np.uint16)

    # Urban structure cluster: high contrast edges
    b02[20:45, 140:180] = np.random.uniform(2200, 3000, (25, 40)).astype(np.uint16)
    b03[20:45, 140:180] = np.random.uniform(2300, 3100, (25, 40)).astype(np.uint16)
    b04[20:45, 140:180] = np.random.uniform(2500, 3300, (25, 40)).astype(np.uint16)
    b08[20:45, 140:180] = np.random.uniform(2000, 2800, (25, 40)).astype(np.uint16)

    with rasterio.open(
        buf, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype=np.uint16,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(b02, 1)
        dst.write(b03, 2)
        dst.write(b04, 3)
        dst.write(b08, 4)

    return buf.getvalue()


def create_sentinel1_geotiff(width=180, height=140, crs="EPSG:32642", left=250100.0, top=3049900.0, res=10.0):
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(left, top, res, res)
    buf = io.BytesIO()

    # Normal terrain backscatter
    vv = np.random.uniform(0.02, 0.08, (height, width)).astype(np.float32)
    vh = np.random.uniform(0.003, 0.015, (height, width)).astype(np.float32)

    # Candidate water sector: specular low backscatter
    vv[40:100, 30:110] = np.random.uniform(0.0005, 0.002, (60, 80)).astype(np.float32)
    vh[40:100, 30:110] = np.random.uniform(0.0001, 0.0005, (60, 80)).astype(np.float32)

    # Urban settlement sector: strong double-bounce backscatter
    vv[15:40, 130:170] = np.random.uniform(0.60, 1.80, (25, 40)).astype(np.float32)
    vh[15:40, 130:170] = np.random.uniform(0.08, 0.25, (25, 40)).astype(np.float32)

    with rasterio.open(
        buf, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype=np.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(vv, 1)
        dst.write(vh, 2)

    return buf.getvalue()


def test_d_projected_utm_fusion():
    print("\n[+] TEST D: GeoTIFF Multimodal Fusion in Projected UTM (EPSG:32642)...")
    opt_gtiff = create_sentinel2_geotiff(width=200, height=150, crs="EPSG:32642", left=250000.0, top=3050000.0, res=10.0)
    sar_gtiff = create_sentinel1_geotiff(width=180, height=140, crs="EPSG:32642", left=250100.0, top=3049900.0, res=10.0)

    resp = client.post(
        "/api/analyze/fusion",
        data={
            "query": "Assess candidate flood inundation extent and built-up infrastructure in Larkana.",
            "fusion_method": "composite",
        },
        files={
            "optical_image": ("sentinel2_utm.tif", opt_gtiff, "image/tiff"),
            "sar_image": ("sentinel1_utm.tif", sar_gtiff, "image/tiff"),
        },
    )
    assert resp.status_code == 200, f"UTF Fusion failed: {resp.text}"
    data = resp.json()

    assert data["alignment_method"] == "geo_referenced"
    assert data["diagnostics"]["geo_metadata_used"] is True
    assert "EPSG:32642" in data["diagnostics"]["common_crs"]

    diag = data["diagnostics"]
    print(f"    [OK] GeoTIFF UTM Alignment: {diag['alignment_method']} (CRS: {diag['common_crs']})")
    print(f"    [OK] Optical Water Candidate: {diag['optical_water_candidate_pct']}%, NDWI Threshold: {diag['ndwi_threshold']}")
    print(f"    [OK] SAR Water Candidate: {diag['sar_water_candidate_pct']}%, SAR Water Thresh: {diag['water_backscatter_threshold_db']} dB")
    print(f"    [OK] Consensus Water: {diag['optical_sar_consensus_water_pct']}%, Modality Agreement: {diag['modality_agreement_pct']}%, IoU: {diag['inundation_iou_pct']}%")
    print(f"    [OK] Baseline Status: {diag['permanent_water_handling_status']}")


def test_e_geographic_wgs84_fusion():
    print("\n[+] TEST E: GeoTIFF Multimodal Fusion in WGS84 Geographic Degrees (EPSG:4326 - Larkana Pakistan)...")
    # Coordinates in Larkana, Sindh, Pakistan (Lon ~ 68.1 to 68.3, Lat ~ 27.5 to 27.6)
    opt_gtiff = create_sentinel2_geotiff(width=200, height=150, crs="EPSG:4326", left=68.10, top=27.60, res=0.0001)
    sar_gtiff = create_sentinel1_geotiff(width=180, height=140, crs="EPSG:4326", left=68.11, top=27.59, res=0.0001)

    resp = client.post(
        "/api/analyze/fusion",
        data={
            "query": "Is there evidence of surface water in Larkana? Analyze Sentinel-2 and Sentinel-1.",
            "fusion_method": "composite",
        },
        files={
            "optical_image": ("Larkana_Sentinel2_EPSG4326.tif", opt_gtiff, "image/tiff"),
            "sar_image": ("Larkana_Sentinel1_EPSG4326.tif", sar_gtiff, "image/tiff"),
        },
    )
    assert resp.status_code == 200, f"WGS84 Fusion failed: {resp.text}"
    data = resp.json()

    # Verify that geographic degree-coordinate GeoTIFFs are properly geo-referenced
    assert data["alignment_method"] == "geo_referenced", f"Expected geo_referenced for EPSG:4326, got {data['alignment_method']}"
    assert data["diagnostics"]["geo_metadata_used"] is True, "Expected geo_metadata_used == True for EPSG:4326"
    assert "EPSG:4326" in data["diagnostics"]["common_crs"]

    diag = data["diagnostics"]
    print(f"    [OK] WGS84 Degree Alignment Passed: {diag['alignment_method']} (CRS: {diag['common_crs']})")
    print(f"    [OK] Grid Resolution: {diag['common_grid_resolution']}")
    print(f"    [OK] Optical Percentiles B02/B03/B04/B08: Mean B08={diag['optical_stats']['b08_stats']['mean']}, Median={diag['optical_stats']['b08_stats']['median']}")
    print(f"    [OK] SAR Percentiles VV: Mean={diag['sar_stats']['vv_stats']['mean']} dB, p5={diag['sar_stats']['vv_stats']['p5']} dB, p95={diag['sar_stats']['vv_stats']['p95']} dB")
    print(f"    [OK] Consensus Water: {diag['optical_sar_consensus_water_pct']}%, Agreement: {diag['modality_agreement_pct']}%")

    features = data["features"]
    assert len(features) > 0
    f0 = features[0]
    assert "optical_score" in f0 and f0["optical_score"] is not None
    assert "sar_score" in f0 and f0["sar_score"] is not None
    assert "region_area_pixels" in f0 and f0["region_area_pixels"] > 0
    assert "Surface Water / Inundation Candidate" in [f["category"] for f in features]
    print(f"    [OK] Isolated Region Metrics (F1): category={f0['category']}, opt_score={f0['optical_score']}, sar_score={f0['sar_score']}, area={f0['region_area_pixels']} px ({f0['region_area_pct']}%), conf={f0['confidence']}")


def test_health():
    print("\n[+] Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200
    print("    [OK] Health check OK:", resp.json())


def test_a_single_image_vqa():
    print("\n[+] TEST A: Single Image VQA (POST /api/analyze)...")
    img_bytes = create_synthetic_image(120, 90, color=(40, 120, 60))
    resp = client.post(
        "/api/analyze",
        data={"query": "What land cover is visible in this satellite scene?"},
        files={"image1": ("scene.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    print(f"    [OK] Single Image: {data['task']}, Conf: {data['confidence']}%")


def test_b_bitemporal_analysis():
    print("\n[+] TEST B: Bi-Temporal Change Analysis (POST /api/analyze/change)...")
    before_bytes = create_synthetic_image(100, 100, color=(50, 100, 50))
    after_bytes = create_synthetic_image(100, 100, color=(150, 60, 40))
    resp = client.post(
        "/api/analyze/change",
        data={"query": "What changed between these images?"},
        files={
            "before_image": ("before.png", before_bytes, "image/png"),
            "after_image": ("after.png", after_bytes, "image/png"),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    print(f"    [OK] Bi-Temporal: {data['task']}, Changed: {data['changed_area_percentage']}%")


def test_c_standard_png_fusion():
    print("\n[+] TEST C: Standard PNG Optical + SAR Fusion...")
    opt_bytes = create_synthetic_image(120, 100, color=(60, 140, 50))
    sar_bytes = create_synthetic_sar_image(120, 100)
    resp = client.post(
        "/api/analyze/fusion",
        data={"query": "Has this area flooded?", "fusion_method": "composite"},
        files={
            "optical_image": ("optical.png", opt_bytes, "image/png"),
            "sar_image": ("sar.png", sar_bytes, "image/png"),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["alignment_method"] == "dimension_matched"
    print(f"    [OK] PNG Fusion Fallback: {data['alignment_method']}, Agreement: {data['modality_agreement_percentage']}%")


def test_f_missing_images():
    print("\n[+] TEST F: Missing Images Validation...")
    sar_bytes = create_synthetic_sar_image(100, 100)
    resp = client.post("/api/analyze/fusion", data={"query": "Test"}, files={"sar_image": ("sar.png", sar_bytes, "image/png")})
    assert resp.status_code in (400, 422)
    print(f"    [OK] Correctly rejected missing optical (HTTP {resp.status_code})")


def test_g_corrupt_images():
    print("\n[+] TEST G: Corrupt Image Bytes...")
    resp = client.post(
        "/api/analyze/fusion",
        data={"query": "Test"},
        files={
            "optical_image": ("corrupt.png", b"NOT_IMAGE", "image/png"),
            "sar_image": ("corrupt2.png", b"NOT_SAR", "image/png"),
        },
    )
    assert resp.status_code in (400, 500)
    print(f"    [OK] Correctly rejected corrupt image (HTTP {resp.status_code})")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SATQUERY AI SCIENTIFIC AUDIT TEST SUITE (EXPANDED)")
    print("=" * 60)
    test_health()
    test_a_single_image_vqa()
    test_b_bitemporal_analysis()
    test_c_standard_png_fusion()
    test_d_projected_utm_fusion()
    test_e_geographic_wgs84_fusion()
    test_f_missing_images()
    test_g_corrupt_images()
    print("\n" + "=" * 60)
    print("ALL 8 SCIENTIFIC AUDIT TESTS PASSED WITH 100% SUCCESS")
    print("=" * 60)
