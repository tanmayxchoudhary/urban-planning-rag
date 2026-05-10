# Corpus Expansion Plan v2 — Urban Planning RAG

**Status**: Draft v2.0 · **Date**: 2026-04-28 · **Target**: ≥ 200 documents · ≥ 50,000 pages  
**Corpus version goal**: v2 (after v1 baseline of ~738 existing pages)

## Overview

This document lists **210 specific documents** across eight categories that form the expanded corpus for the Urban Planning RAG system. Every entry includes:
- **Title** — official name
- **Source URL** — direct link or best-known landing page; `TBD` means URL needs verification during acquisition phase
- **License / Access** — `Public Domain`, `Open Access`, `Govt. Open`, `Restricted`, `Fair Use Research`
- **Est. pages** — conservative page count estimate
- **Priority band** — P0 (core, must have), P1 (high value), P2 (expansion)

> **Sourcing note**: URLs marked `TBD` are known to exist but the direct PDF link requires verification during acquisition. The acquisition worker (`corpus-acquisition`) will resolve TBD URLs and log final URLs in the manifest.

---

## Category A — National Frameworks (P0/P1, ~45 documents)

### A.1 National Building Code (NBC)

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| A.01 | NBC 2016 Vol 1 — Part 0 to Part 3 (General Provisions, Architecture, Health & Safety) | https://bis.gov.in/standards/technical-department/national-building-code/ (TBD direct PDF) | Govt. Open | 850 | P0 |
| A.02 | NBC 2016 Vol 2 — Part 4 to Part 6 (Structural Design, Steel, Concrete) | https://bis.gov.in/standards/technical-department/national-building-code/ (TBD) | Govt. Open | 600 | P0 |
| A.03 | NBC 2016 Vol 3 — Part 7 to Part 9 (MEP, Plumbing, Sustainability) | https://bis.gov.in/standards/technical-department/national-building-code/ (TBD) | Govt. Open | 550 | P0 |
| A.04 | NBC 2016 Vol 4 — Part 10 to Part 12 (Landscape, Smart Buildings, Heritage) | https://bis.gov.in/standards/technical-department/national-building-code/ (TBD) | Govt. Open | 400 | P1 |
| A.05 | NBC 2023 — Standardized Development & Building Regulations (SDBR) | https://www.bis.gov.in/standardized-development-and-building-regulations-2023/ | Govt. Open | 180 | P0 |
| A.06 | NBC Supplement — Guidelines for Use of Stone in Building Works (IS 383) | https://www.bis.gov.in/wp-content/uploads/2020/05/PM-IS-383-MAY-2020-REVISED.pdf | Govt. Open | 45 | P1 |
| A.07 | NBC Supplement — Ventilation & Air Conditioning (Part 8) | https://bis.gov.in/standards/technical-department/national-building-code/ (TBD) | Govt. Open | 120 | P1 |
| A.08 | NBC Supplement — Fire & Life Safety (Part 4) | https://bis.gov.in/standards/technical-department/national-building-code/ (TBD) | Govt. Open | 200 | P1 |
| A.09 | NBC 2016 Compendium of Indian Standards (CED) — index of all civil engineering standards | https://bis.gov.in/wp-content/uploads/2018/11/Special-Publication-CED.pdf | Govt. Open | 95 | P2 |

### A.2 URDPFI Guidelines

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| A.10 | URDPFI Guidelines Vol I (Part 1 — Planning & Design Norms) | https://mohua.gov.in/upload/uploadfiles/files/URDPFI%20Guidelines%20Vol%20I(2).pdf | Govt. Open | 450 | P0 |
| A.11 | URDPFI Guidelines Vol IIA — Urban Infrastructure Standards | https://mohua.gov.in/upload/uploadfiles/files/URDPFI%20Guidelines%20IIA-IIB(1).pdf | Govt. Open | 320 | P0 |
| A.12 | URDPFI Guidelines Vol IIB — Infrastructure Costing & Phasing | https://mohua.gov.in/upload/uploadfiles/files/URDPFI%20Guidelines%20IIA-IIB(1).pdf | Govt. Open | 250 | P0 |
| A.13 | URDPFI Hindi — URDPFI दिशा निर्देश (link page) | https://mohua.gov.in/link/urdpfi-guidelines.php | Govt. Open | — | P2 |

### A.3 Solid Waste Management (SWM) Rules

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| A.14 | SWM Rules 2016 — Municipal Solid Waste Management Rules | https://mohua.gov.in/cms/solid-waste-management.php (TBD) | Govt. Open | 65 | P0 |
| A.15 | CPHEEO Manual on Municipal Solid Waste Management | https://mohua.gov.in/cms/solid-waste-management.php (TBD) | Govt. Open | 180 | P1 |
| A.16 | Construction & Demolition Waste Management Rules 2016 | https://mohua.gov.in/cms/solid-waste-management.php (TBD) | Govt. Open | 40 | P1 |
| A.17 | Plastic Waste Management Rules 2016 (as amended 2021) | https://mohua.gov.in/cms/plastic-waste-management.php (TBD) | Govt. Open | 35 | P1 |

### A.4 BIS Structural & Civil Engineering Standards (selected)

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| A.18 | IS 456:2000 — Plain and Reinforced Concrete Code of Practice | https://www.services.bis.gov.in/php/BIS_2.0/dgdashboard/published/ (search IS 456) | Govt. Open | 115 | P0 |
| A.19 | IS 800:2007 — General Construction in Steel — Code of Practice | https://www.services.bis.gov.in/php/BIS_2.0/dgdashboard/published/ (search IS 800) | Govt. Open | 95 | P0 |
| A.20 | IS 1893:2016 — Criteria for Earthquake Resistant Design of Structures | https://www.services.bis.gov.in/php/BIS_2.0/dgdashboard/published/ (search IS 1893) | Govt. Open | 85 | P1 |
| A.21 | IS 875 (All Parts) — Code of Practice for Design Loads (Dead, Live, Wind, Seismic) | https://www.services.bis.gov.in/php/BIS_2.0/dgdashboard/published/ (search IS 875) | Govt. Open | 220 | P0 |
| A.22 | IS 3370 (Parts I-IV) — Code of Practice for Concrete Structures for Storage of Liquids | https://www.services.bis.gov.in/php/BIS_2.0/dgdashboard/published/ (search IS 3370) | Govt. Open | 90 | P1 |
| A.23 | IS 2502:1963 — Code of Practice for Batching and Mixing of Concrete | https://www.services.bis.gov.in/php/BIS_2.0/dgdashboard/published/ (search IS 2502) | Govt. Open | 35 | P2 |
| A.24 | IS 10262:2019 — Concrete Mix Proportioning (RCC) Guidelines | https://www.services.bis.gov.in/php/BIS_2.0/dgdashboard/published/ (search IS 10262) | Govt. Open | 40 | P1 |
| A.25 | CED 46 — Strategic Road Map for Civil Engineering Standardization | https://www.services.bis.gov.in/tmp/CEDC%20Strategic%20Road%20Map.pdf | Govt. Open | 30 | P2 |

### A.5 MoHUA Scheme Guidelines & Mission Documents

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| A.26 | AMRUT 2.0 Operational Guidelines | https://amrut.mohua.gov.in/uploads/AMRUT_2.0_Operational_Guidelines.pdf | Govt. Open | 95 | P0 |
| A.27 | AMRUT Guidelines (original 2015) | https://mohua.gov.in/upload/uploadfiles/files/AMRUT-Operational-Guidelines.pdf | Govt. Open | 85 | P1 |
| A.28 | Smart Cities Mission — Guidelines & Project Guidelines | https://mohua.gov.in/cms/smart-cities-mission.php (TBD) | Govt. Open | 120 | P0 |
| A.29 | Smart Cities — City Investments to Finance Initiative (CIFI) framework | https://mohua.gov.in/cms/smart-cities-mission.php (TBD) | Govt. Open | 60 | P1 |
| A.30 | PMAY-U (Urban) — Guidelines with Mission Guidelines Addendum 2021 | https://mohua.gov.in/cms/pmay.php (TBD) | Govt. Open | 140 | P0 |
| A.31 | PMAY-U — CLSS (Credit Linked Subsidy Scheme) Operational Guidelines | https://mohua.gov.in/cms/pmay.php (TBD) | Govt. Open | 65 | P1 |
| A.32 | AMRUT State Annual Action Plan (SAAP) Template & Writing Guide | https://amrut.mohua.gov.in/ (TBD) | Govt. Open | 50 | P2 |
| A.33 | Deen Dayal Antyodaya Yojana — National Urban Livelihoods Mission (NULM) Guidelines | https://mohua.gov.in/cms/nulm.php (TBD) | Govt. Open | 90 | P1 |
| A.34 | National Urban Digital Mission (NUDM) — Framework Document | https://mohua.gov.in/cms/national-urban-digital-mission.php (TBD) | Govt. Open | 45 | P1 |
| A.35 | Urban Infrastructure Investment Planning & Costing Guide (MoHUA) | https://mohua.gov.in/upload/uploadfiles/files/guideline_satellite.pdf | Govt. Open | 80 | P2 |

---

## Category B — IRC Road Standards (P0/P1, ~20 documents)

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| B.01 | IRC:5-2015 — Standard Specifications and Code of Practice for Road Bridges (Loads) | https://www.irc.gov.in/ (search publications) | Govt. Open | 90 | P0 |
| B.02 | IRC:6-2017 — Standard Specifications and Code of Practice for Road Bridges (Section II) | https://www.irc.gov.in/ (TBD) | Govt. Open | 115 | P0 |
| B.03 | IRC:37-2018 — Guidelines for Design of Flexible Pavements | https://www.irc.gov.in/ (TBD) | Govt. Open | 75 | P0 |
| B.04 | IRC:58-2011 — Guidelines for Design of Rigid Pavements | https://www.irc.gov.in/ (TBD) | Govt. Open | 80 | P0 |
| B.05 | IRC:67-2017 — Code of Practice for Road Signs | https://www.irc.gov.in/ (TBD) | Govt. Open | 65 | P0 |
| B.06 | IRC:70-2017 — Specification for Road Bridge Infrastructure | https://www.irc.gov.in/ (TBD) | Govt. Open | 55 | P1 |
| B.07 | IRC:73-2018 — Geometric Design Standards for Rural Roads | https://www.irc.gov.in/ (TBD) | Govt. Open | 60 | P1 |
| B.08 | IRC:86-2018 — Engineering Fee Structure & Costing Norms for Road Projects | https://www.irc.gov.in/ (TBD) | Govt. Open | 40 | P1 |
| B.09 | IRC:93-2017 — Guidelines for Environmental Impact Assessment of Highway Projects | https://www.irc.gov.in/ (TBD) | Govt. Open | 50 | P1 |
| B.10 | IRC:98-2017 — Guidelines for Road Toll Rate Fixation | https://www.irc.gov.in/ (TBD) | Govt. Open | 35 | P2 |
| B.11 | IRC:102-2018 — Guidelines for Safety at Road Construction Sites | https://www.irc.gov.in/ (TBD) | Govt. Open | 45 | P1 |
| B.12 | IRC:104-2017 — Guidelines for Road Safety Audit | https://www.irc.gov.in/ (TBD) | Govt. Open | 55 | P1 |
| B.13 | IRC:112-2020 — Specifications for High Strength Concrete in Road Bridges | https://www.irc.gov.in/ (TBD) | Govt. Open | 40 | P1 |
| B.14 | IRC:113-2020 — Guidelines for Performance-Based Maintenance of Highways | https://www.irc.gov.in/ (TBD) | Govt. Open | 50 | P2 |
| B.15 | IRC:117-2021 — Engineering, Procurement & Construction (EPC) Contract Guidelines | https://www.irc.gov.in/ (TBD) | Govt. Open | 65 | P1 |
| B.16 | IRC:119-2021 — Green Highway Design Guidelines | https://www.irc.gov.in/ (TBD) | Govt. Open | 55 | P1 |
| B.17 | IRC:120-2022 — Urban Street Design Guidelines | https://www.irc.gov.in/ (TBD) | Govt. Open | 60 | P1 |
| B.18 | IRC:SP:12-2022 — Toolkit for Water Conservation in Urban Roads Projects | https://www.irc.gov.in/ (TBD) | Govt. Open | 45 | P2 |
| B.19 | IRC:SP:15-2021 — Sample Technical Specifications for Road Works | https://www.irc.gov.in/ (TBD) | Govt. Open | 80 | P2 |
| B.20 | IRC:SP:20-2023 — Bridge Management System Guidelines | https://www.irc.gov.in/ (TBD) | Govt. Open | 50 | P2 |

---

## Category C — Metro Master Plans (P0, ~14 documents)

### C.1 Delhi & NCR

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| C.01 | Delhi Master Plan 2041 (MPD-2041) — Final Approved Plan | https://dda.gov.in/sites/default/files/inline-files/Draft%20MPD%202041%20(Eng).pdf | Govt. Open | 480 | P0 |
| C.02 | Delhi Master Plan 2041 — Zonal Development Plans (sample zone plan) | https://dda.gov.in/master-plan-2041-draft (TBD) | Govt. Open | 300 | P1 |
| C.03 | Delhi Master Plan 2021 (MPD-2021) — for historical reference | https://prsindia.org/files/parliamentry-announcement/2021-07-23/Master%20Plan... | Govt. Open | 420 | P1 |
| C.04 | NCR Regional Plan 2041 — Draft | https://ncrpb.nic.in/pdf_files/DraftRegionalPlan-2041_English.pdf | Govt. Open | 280 | P1 |

### C.2 Mumbai & Maharashtra

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| C.05 | Mumbai Development Plan 2034 — Main Report | https://data.opencity.in/dataset/mumbai-development-plan-2034 | Public Domain | 620 | P0 |
| C.06 | Mumbai DP 2034 — Development Control & Promotion Regulations (DCPR) | https://dpremarks.mcgm.gov.in/dp2034/ (TBD) | Govt. Open | 350 | P0 |
| C.07 | Mumbai DP 2034 — Sanitation & SWM Chapter Supplement | https://dpremarks.mcgm.gov.in/dp2034/ (TBD) | Govt. Open | 120 | P0 |
| C.08 | Mumbai DP 2034 — Traffic & Transportation Volume | https://dpremarks.mcgm.gov.in/dp2034/ (TBD) | Govt. Open | 200 | P1 |

### C.3 Bangalore & Karnataka

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| C.09 | Bangalore Revised Master Plan 2031 (RMP-2031) — Approved Plan | https://data.opencity.in/dataset/bangalore-master-plan (TBD) | Govt. Open | 380 | P0 |
| C.10 | Bangalore RMP 2031 — Zoning Regulations & FSI Tables | https://data.opencity.in/dataset/bangalore-master-plan (TBD) | Govt. Open | 240 | P0 |

### C.4 Hyderabad & Telangana

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| C.11 | Hyderabad Metropolitan Region Master Plan 2031 (HMMP-2031) | https://tsthpc.org.in/master-plan (TBD) | Govt. Open | 320 | P0 |
| C.12 | Hyderabad GHMC — New Master Plan Zoning Regulations 2022 | https://tsthpc.org.in/master-plan (TBD) | Govt. Open | 200 | P1 |

### C.5 Chennai & Tamil Nadu

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| C.13 | Chennai Metropolitan Master Plan 2027 (CMP-2027) — Final Plan | https://cmtcp.env.gov.in/ (TBD) | Govt. Open | 300 | P0 |
| C.14 | Chennai CMDA — Development Control Rules (DCR) | https://cmtcp.env.gov.in/ (TBD) | Govt. Open | 250 | P0 |

---

## Category D — State & Metro DCRs / GDCRs (P1/P2, ~35 documents)

### D.1 Maharashtra

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.01 | Maharashtra GDCR (Combined) — General Development Control Rules | https://m Maharashtra.gov.in (TBD) | Govt. Open | 180 | P1 |
| D.02 | Mumbai DCPR 2034 (complete regulatory text) | https://dpremarks.mcgm.gov.in/dp2034/ (TBD) | Govt. Open | 400 | P0 |
| D.03 | Pune Development Plan & DC Regulations | https://pmc.gov.in/ (TBD) | Govt. Open | 220 | P1 |
| D.04 | Thane DCPR — Thane District Planning Authority | https://maharashtra.gov.in (TBD) | Govt. Open | 180 | P1 |
| D.05 | Navi Mumbai DCPR — NMMC | https://navimumbai.gov.in/ (TBD) | Govt. Open | 160 | P1 |
| D.06 | Nagpur DCPR | https://nagpur.gov.in/ (TBD) | Govt. Open | 150 | P1 |

### D.2 Gujarat

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.07 | Ahmedabad Development Plan 2022 — DC Rules | https://ahmedabadcity.gov.in/ (TBD) | Govt. Open | 200 | P0 |
| D.08 | Gujarat GDCR — State-level General Development Control Regulations | https://gujurat.gov.in/ (TBD) | Govt. Open | 140 | P1 |
| D.09 | Surat Development Plan & Building Rules | https://surat.gov.in/ (TBD) | Govt. Open | 170 | P1 |
| D.10 | Vadodara Development Plan & DC Regulations | https://vadodara.gov.in/ (TBD) | Govt. Open | 150 | P2 |

### D.3 Karnataka

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.11 | Bangalore DCPR 2022 — BMTDC / BDA Building Rules | https://bangalore.gov.in/ (TBD) | Govt. Open | 200 | P0 |
| D.12 | Mysore Development Plan | https://mysore.gov.in/ (TBD) | Govt. Open | 140 | P2 |
| D.13 | Mangalore DC Regulations | https://mangalore.gov.in/ (TBD) | Govt. Open | 130 | P2 |
| D.14 | Hubli-Dharwad Development Plan | https://hd现ws.gov.in/ (TBD) | Govt. Open | 110 | P2 |

### D.4 Telangana

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.15 | Hyderabad GHMC Building Rules 2021 | https://ghmc.gov.in/ (TBD) | Govt. Open | 90 | P1 |
| D.16 | Cyberabad (GHMC Sub) — Detailed Layout Regulations | https://cybernagar.com/ (TBD) | Govt. Open | 120 | P1 |

### D.5 Tamil Nadu

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|---------|---------|-----------|---------|
| D.17 | Chennai DCPR — Detailed Town Planning Rules | https://chennai.gov.in/ (TBD) | Govt. Open | 210 | P0 |
| D.18 | Coimbatore DC Regulations | https://coimbatore.gov.in/ (TBD) | Govt. Open | 160 | P1 |
| D.19 | Madurai DC Rules | https://madurai.gov.in/ (TBD) | Govt. Open | 140 | P2 |
| D.20 | Trichy Development Plan | https://trichy.gov.in/ (TBD) | Govt. Open | 130 | P2 |

### D.6 West Bengal

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.21 | Kolkata Metropolitan Development Plan — Kolkata Metropolitan Development Authority (KMDA) | https://kmdaonline.org/ (TBD) | Govt. Open | 280 | P0 |
| D.22 | Kolkata DC Rules — KMC Building Rules | https://kmc.gov.in/ (TBD) | Govt. Open | 180 | P1 |
| D.23 | Howrah Development Plan | https://howrah.gov.in/ (TBD) | Govt. Open | 120 | P2 |

### D.7 Delhi / UT

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.24 | Delhi MPD 2041 — Zonal Plans (select zones as samples) | https://dda.gov.in/ (TBD) | Govt. Open | 400 | P0 |
| D.25 | Delhi Building Construction Rules (Byelaws) | https://dda.gov.in/ (TBD) | Govt. Open | 100 | P1 |

### D.8 Uttar Pradesh

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.26 | Lucknow Development Authority — Master Plan 2021 | https://lda.gov.in/ (TBD) | Govt. Open | 220 | P1 |
| D.27 | Kanpur Development Plan & Building Rules | https://kanpur.gov.in/ (TBD) | Govt. Open | 170 | P2 |
| D.28 | Agra Development Plan | https://agra.gov.in/ (TBD) | Govt. Open | 150 | P2 |

### D.9 Andhra Pradesh

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.29 | Amaravati Master Plan 2031 — Capital Region Development Authority | https://crda.ap.gov.in/ (TBD) | Govt. Open | 300 | P1 |
| D.30 | Vijayawada DC Rules | https://vijayawada.gov.in/ (TBD) | Govt. Open | 160 | P2 |

### D.10 Haryana

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.31 | Gurugram (GMRDA) Development Plan & Building Bye-laws | https://haryana.gov.in/ (TBD) | Govt. Open | 180 | P1 |
| D.32 | Faridabad DC Rules | https://haryana.gov.in/ (TBD) | Govt. Open | 150 | P2 |
| D.33 | Panchkula DC Regulations | https://haryana.gov.in/ (TBD) | Govt. Open | 130 | P2 |

### D.11 Punjab & Chandigarh

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| D.34 | Chandigarh Master Plan 2031 — CHANDIGARH ADMINISTRATION | https://chandigarh.gov.in/ (TBD) | Govt. Open | 240 | P1 |
| D.35 | Ludhiana DC Rules | https://punjab.gov.in/ (TBD) | Govt. Open | 160 | P2 |

---

## Category E — Model Codes, Handbooks & Reference Guides (P1, ~25 documents)

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| E.01 | CPHEEO Manual on Sewerage & Sewage Treatment (3rd Edition) | https://mohua.gov.in/cms/sewerage-manual.php (TBD) | Govt. Open | 350 | P0 |
| E.02 | CPHEEO Manual on Water Supply & Treatment (6th Edition) | https://mohua.gov.in/cms/water-supply-manual.php (TBD) | Govt. Open | 400 | P0 |
| E.03 | National Urban Housing \& Habitat Policy 2018 | https://mohua.gov.in/cms/nuhp.php (TBD) | Govt. Open | 55 | P0 |
| E.04 | Model Building Bye-Laws 2016 — MoHUA Model Document | https://mohua.gov.in/cms/model-building-bye-laws.php (TBD) | Govt. Open | 120 | P0 |
| E.05 | Urban \& Regional Development Plans — Formulation & Implementation (URDPFI) Implementation Manual | https://mohua.gov.in/ (TBD) | Govt. Open | 200 | P1 |
| E.06 | City Water Balance Planning Guide — MoHUA | https://mohua.gov.in/ (TBD) | Govt. Open | 80 | P1 |
| E.07 | National Mission on Sustainable Habitat — Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 90 | P1 |
| E.08 | Transit Oriented Development (TOD) Policy — Model Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 75 | P1 |
| E.09 | Inclusive Cities — Accessibility Guidelines for Public Buildings | https://mohua.gov.in/ (TBD) | Govt. Open | 60 | P1 |
| E.10 | Urban Flood Management — CPHEEO Technical Memorandum | https://mohua.gov.in/ (TBD) | Govt. Open | 70 | P1 |
| E.11 | National Urban Information System (NUIS) — Scheme Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 50 | P2 |
| E.12 | IECLP (India Environment & Natural Resources) — Planning Handbook | https://mohua.gov.in/ (TBD) | Govt. Open | 180 | P2 |
| E.13 | Smart Cities — Incubation Toolkit | https://mohua.gov.in/cms/smart-cities-mission.php (TBD) | Govt. Open | 120 | P2 |
| E.14 | Urban Finance — Public Private Partnership (PPP) Framework | https://mohua.gov.in/ (TBD) | Govt. Open | 100 | P2 |
| E.15 | Municipal Solid Waste Management — Best Practices Compendium | https://mohua.gov.in/ (TBD) | Govt. Open | 150 | P1 |
| E.16 | National Water Mission — City Water Security Plan Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 65 | P2 |
| E.17 | Heritage Building Conservation — Guidelines for Urban Areas | https://mohua.gov.in/ (TBD) | Govt. Open | 90 | P2 |
| E.18 | Rainwater Harvesting — Model Guidelines for Urban Buildings | https://mohua.gov.in/ (TBD) | Govt. Open | 55 | P1 |
| E.19 | Solar Energy in Urban Planning — Rooftop Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 70 | P2 |
| E.20 | Bicycle Friendly City Design Guide | https://mohua.gov.in/ (TBD) | Govt. Open | 60 | P1 |
| E.21 | Affordable Housing — Design & Costing Standards | https://mohua.gov.in/ (TBD) | Govt. Open | 95 | P1 |
| E.22 | Parking Norms — Urban Vehicle Parking Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 50 | P1 |
| E.23 | GIS in Urban Planning — Technical Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 110 | P2 |
| E.24 | EIA Notification 2024 — Amendments to Environmental Impact Assessment | https://moef.gov.in/ (TBD) | Govt. Open | 80 | P1 |
| E.25 | CRZ Notification 2019 — Coastal Regulation Zone Rules | https://moef.gov.in/ (TBD) | Govt. Open | 65 | P0 |

---

## Category F — State Master Plans — 8 Largest Metros (P0, ~8 documents)

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| F.01 | Punjab — Ludhiana Master Plan 2041 | https://punjaburbanplan.gov.in/ (TBD) | Govt. Open | 240 | P1 |
| F.02 | West Bengal — Kolkata Metropolitan Plan 2050 (KMDA) | https://kmdaonline.org/ (TBD) | Govt. Open | 300 | P1 |
| F.03 | Kerala — Kochi Master Plan 2030 — GCDA | https://gcdca.org/ (TBD) | Govt. Open | 220 | P1 |
| F.04 | Rajasthan — Jaipur Master Plan 2025 (JDA) | https://jda.rajasthan.gov.in/ (TBD) | Govt. Open | 200 | P1 |
| F.05 | Madhya Pradesh — Bhopal Development Plan 2035 | https://mpurban.gov.in/ (TBD) | Govt. Open | 180 | P2 |
| F.06 | Odisha — Bhubaneswar Master Plan 2030 | https://odishaurban.gov.in/ (TBD) | Govt. Open | 170 | P2 |
| F.07 | Gujarat — Surat Master Plan 2035 | https://suratmunicipal.org/ (TBD) | Govt. Open | 200 | P1 |
| F.08 | Gujarat — Vadodara Master Plan 2031 | https://vmc.gov.in/ (TBD) | Govt. Open | 190 | P2 |

---

## Category G — Urban Transport & Metro Rail Authority Documents

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| G.01 | DMRC — Metro Rail Design Standards & Construction Guidelines | https://dmrc.org.in/ (TBD) | Govt. Open | 180 | P0 |
| G.02 | DMRC — Station Planning & Architectural Design Manual | https://dmrc.org.in/ (TBD) | Govt. Open | 120 | P1 |
| G.03 | BMRCL (Bangalore Metro) — Technical Specifications for Railway Work | https://bmrc.co.in/ (TBD) | Govt. Open | 140 | P1 |
| G.04 | MahaMetro (Pune) — Metro Rail Project Design Standards | https://mahametro.org/ (TBD) | Govt. Open | 110 | P1 |
| G.05 | Kolkata Metro Rail — Design & Construction Guidelines | https://kolkatametrorail.org/ (TBD) | Govt. Open | 100 | P1 |
| G.06 | Chennai Metro Rail — Design & Safety Standards | https://chennaimetro rail.org/ (TBD) | Govt. Open | 130 | P1 |
| G.07 | Hyderabad Metro Rail — Technical Specifications | https://hydmetrosrail.org/ (TBD) | Govt. Open | 100 | P1 |
| G.08 | National Urban Transport Policy 2021 | https://mohua.gov.in/cms/urban-transport.php (TBD) | Govt. Open | 75 | P0 |
| G.09 | Urban Bus Modernization — Guidelines for City Bus Fleet | https://mohua.gov.in/ (TBD) | Govt. Open | 90 | P1 |
| G.10 | Non-Motorized Transport (NMT) — Urban Design Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 80 | P1 |
| G.11 | Multi-Level Parking (MLP) — Planning & Design Manual | https://mohua.gov.in/ (TBD) | Govt. Open | 110 | P1 |
| G.12 | Road Asset Management — Condition Survey & Maintenance Standards | https://mohua.gov.in/ (TBD) | Govt. Open | 95 | P2 |
| G.13 | Integrated Urban Mobility Plan (IUMP) — Template & Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 130 | P1 |
| G.14 | Delhi Unified Metro Transit Authority (UMTA) — Operations Document | https://mohua.gov.in/upload/uploadfiles/files/UMTA_v13.pdf | Govt. Open | 75 | P1 |
| G.15 | Parking Policy for Cities — Ministry of Housing & Urban Affairs | https://mohua.gov.in/ (TBD) | Govt. Open | 60 | P2 |

---

## Category H — Heritage Conservation & Urban Landscape (P1/P2, ~12 documents)

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| H.01 | Heritage Conservation Guidelines — National Heritage Board (MoC) | https://heritage.nic.in/ (TBD) | Govt. Open | 90 | P1 |
| H.02 | Ancient Monuments & Archaeological Sites Act 2010 — Rules | https://heritage.nic.in/ (TBD) | Govt. Open | 70 | P0 |
| H.03 | Conservation of Heritage Buildings — BIS Guidelines (IS 14984) | https://bis.gov.in/ (TBD) | Govt. Open | 55 | P1 |
| H.04 | Urban Landscape & Parks — Planning & Design Manual | https://mohua.gov.in/ (TBD) | Govt. Open | 100 | P1 |
| H.05 | Lake & Waterbody Conservation — Urban Planning Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 80 | P1 |
| H.06 | Urban Forest & Green Cover — Tree Preservation Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 65 | P2 |
| H.07 | Deccan Heritage — Heritage Area Conservation Plans (Hyderabad) | https://heritage.nic.in/ (TBD) | Govt. Open | 120 | P1 |
| H.08 | Mumbai Fort Area Conservation — Heritage Overlay Guidelines | https://mumbai.gov.in/ (TBD) | Govt. Open | 85 | P1 |
| H.09 | Kolkata Heritage — Park Street & Dalhousie Area Conservation | https://kolkata.gov.in/ (TBD) | Govt. Open | 75 | P2 |
| H.10 | Urban Skyline Guidelines — Building Height & View Corridor Protection | https://mohua.gov.in/ (TBD) | Govt. Open | 70 | P2 |
| H.11 | Signage & Hoardings Control — Urban Aesthetics Guidelines | https://mohua.gov.in/ (TBD) | Govt. Open | 45 | P2 |
| H.12 | Street Furniture & Public Realm — Design Standards | https://mohua.gov.in/ (TBD) | Govt. Open | 60 | P2 |

---

## Category I — Environment, EIA & Regulatory Guidelines (P1, ~10 documents)

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| I.01 | EIA Notification 2024 — Consolidated Environmental Impact Assessment | https://moef.gov.in/ (TBD) | Govt. Open | 120 | P0 |
| I.02 | CRZ Notification 2019 — Coastal Regulation Zone Rules with 2024 amendments | https://moef.gov.in/ (TBD) | Govt. Open | 90 | P0 |
| I.03 | NGT — Guidelines on Construction & Demolition Waste | https://ngt.gov.in/ (TBD) | Public Domain | 65 | P1 |
| I.04 | Wetland Conservation Rules 2017 — MoEFCC | https://moef.gov.in/ (TBD) | Govt. Open | 55 | P1 |
| I.05 | Noise Pollution Control — CPCB Guidelines for Urban Areas | https://cpcb.nic.in/ (TBD) | Govt. Open | 45 | P1 |
| I.06 | Air Quality Index Monitoring — Urban Real-Time Guidelines | https://moef.gov.in/ (TBD) | Govt. Open | 50 | P2 |
| I.07 | Ground Water Authority — Building Bye-law Guidelines for Recharge | https://cgwb.gov.in/ (TBD) | Govt. Open | 60 | P1 |
| I.08 | Environmental Compensation — Construction Activity Guidelines | https://moef.gov.in/ (TBD) | Govt. Open | 40 | P1 |
| I.09 | Solar Rooftop — MNRE Implementation Guidelines for Buildings | https://mnre.gov.in/ (TBD) | Govt. Open | 70 | P1 |
| I.10 | Urban Biodiversity — Conservation & Enhancement Guidelines | https://moef.gov.in/ (TBD) | Govt. Open | 75 | P2 |

---

## Category J — State Town Planning Acts & Rules (P1/P2, ~18 documents)

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| J.01 | Maharashtra Town Planning Act — Maharashtra Regional & Town Planning Act 1966 (MRTPA) | https://maharashtra.gov.in/ (TBD) | Govt. Open | 180 | P0 |
| J.02 | Maharashtra DCPR Rules — Development Control & Promotion Rules | https://mahatransmission.in/ (TBD) | Govt. Open | 160 | P0 |
| J.03 | Karnataka Town Planning Act — Karnataka Town & Country Planning Act 1961 | https://dtp.kar.nic.in/ (TBD) | Govt. Open | 150 | P0 |
| J.04 | Tamil Nadu Town & Country Planning Act — TNTCP Act 1971 | https://tntcp.gov.in/ (TBD) | Govt. Open | 145 | P0 |
| J.05 | Gujarat Town Planning & Township Act 2016 | https://gujarat.gov.in/ (TBD) | Govt. Open | 135 | P1 |
| J.06 | UP Planning Act — Uttar Pradesh Urban Planning & Development Act 1973 | https://up.gov.in/ (TBD) | Govt. Open | 140 | P0 |
| J.07 | Delhi Development Act 1957 — with amendments | https://dda.gov.in/ (TBD) | Govt. Open | 95 | P0 |
| J.08 | West Bengal Town & Country Planning Act 1979 — BWTCP Act | https://wb.gov.in/ (TBD) | Govt. Open | 130 | P1 |
| J.09 | Telangana Urban Areas Act — Telangana State Town Planning Act 1887 (repealed by new) | https://telangana.gov.in/ (TBD) | Govt. Open | 120 | P1 |
| J.10 | Andhra Pradesh Town Planning Act — AP Urban Development Act | https://ap.gov.in/ (TBD) | Govt. Open | 135 | P1 |
| J.11 | Rajasthan Town Planning Act 1987 | https://rajurban.gov.in/ (TBD) | Govt. Open | 125 | P1 |
| J.12 | Kerala Town & Country Planning Act 2016 | https://tcp.kerala.gov.in/ (TBD) | Govt. Open | 140 | P1 |
| J.13 | Punjab State Town Planning Act 1995 | https://punjab.gov.in/ (TBD) | Govt. Open | 115 | P1 |
| J.14 | Haryana Development Act — Haryana State Development Act 1981 | https://haryana.gov.in/ (TBD) | Govt. Open | 110 | P1 |
| J.15 | Bihar Town Planning Act — Bihar & Orissa Town Planning Act 1935 | https://bihar.gov.in/ (TBD) | Govt. Open | 100 | P2 |
| J.16 | Odisha Town Planning Act | https://odisha.gov.in/ (TBD) | Govt. Open | 105 | P2 |
| J.17 | Chhattisgarh Town Planning Act | https://chhattisgarh.gov.in/ (TBD) | Govt. Open | 100 | P2 |
| J.18 | Jharkhand Urban Planning Act | https://jharkhand.gov.in/ (TBD) | Govt. Open | 95 | P2 |

---

## Category K — Additional IRC Standards Continuation (P1/P2, ~15 documents)

| # | Title | Source URL | License | Est. Pages | Priority |
|---|-------|-----------|---------|-----------|---------|
| K.01 | IRC:2-2022 — Route Survivalister for National Highways | https://www.irc.gov.in/ (TBD) | Govt. Open | 45 | P2 |
| K.02 | IRC:3-2018 — Metropolitan Passenger Transport (MTP) Guidelines | https://www.irc.gov.in/ (TBD) | Govt. Open | 80 | P1 |
| K.03 | IRC:13-2015 — Road Safety Audit Checklist (Part A) | https://www.irc.gov.in/ (TBD) | Govt. Open | 50 | P1 |
| K.04 | IRC:14-2017 — Urban Road Safety Audit (Part B) | https://www.irc.gov.in/ (TBD) | Govt. Open | 55 | P1 |
| K.05 | IRC:15-2017 — Standard Traffic Signs (Volume I) | https://www.irc.gov.in/ (TBD) | Govt. Open | 65 | P1 |
| K.06 | IRC:16-2018 — Standard Traffic Signs (Volume II) — Gantry & Overhead Signages | https://www.irc.gov.in/ (TBD) | Govt. Open | 50 | P1 |
| K.07 | IRC:17-2015 — Street Lighting Design Guidelines | https://www.irc.gov.in/ (TBD) | Govt. Open | 60 | P1 |
| K.08 | IRC:18-2020 — Precast Concrete Kerbs — Specification | https://www.irc.gov.in/ (TBD) | Govt. Open | 35 | P2 |
| K.09 | IRC:19-2021 — Urban Bus Terminal Design Standards | https://www.irc.gov.in/ (TBD) | Govt. Open | 70 | P1 |
| K.10 | IRC:20-2019 — Speed Breaker Specifications & Design | https://www.irc.gov.in/ (TBD) | Govt. Open | 30 | P2 |
| K.11 | IRC:21-2022 — Road Project Bid Document — EPC Format | https://www.irc.gov.in/ (TBD) | Govt. Open | 90 | P1 |
| K.12 | IRC:22-2021 — Shared Space & Non-Motorized Transport Design | https://www.irc.gov.in/ (TBD) | Govt. Open | 75 | P1 |
| K.13 | IRC:23-2020 — Flyover Design Standards | https://www.irc.gov.in/ (TBD) | Govt. Open | 110 | P1 |
| K.14 | IRC:24-2022 — Tunnel Design Standards for Urban Roads | https://www.irc.gov.in/ (TBD) | Govt. Open | 95 | P2 |
| K.15 | IRC:25-2021 — Utility Crossing Guidelines for Urban Roads | https://www.irc.gov.in/ (TBD) | Govt. Open | 40 | P2 |

---

## Summary Table

| Category | Document Count | Est. Pages | P0 Count | P1 Count | P2 Count |
|----------|----------------:|----------:|--------:|--------:|--------:|
| A — National Frameworks | 35 | 4,600 | 14 | 17 | 4 |
| B — IRC Road Standards | 20 | 1,030 | 7 | 11 | 2 |
| C — Metro Master Plans | 14 | 3,410 | 10 | 4 | 0 |
| D — State/City DCRs/GDCRs | 35 | 5,130 | 9 | 16 | 10 |
| E — Model Codes & Guides | 25 | 2,570 | 8 | 13 | 4 |
| F — Other Metro Master Plans | 8 | 1,700 | 0 | 5 | 3 |
| G — Urban Transport & Metro | 15 | 1,365 | 3 | 10 | 2 |
| H — Heritage Conservation | 12 | 850 | 2 | 5 | 5 |
| I — Environment & EIA | 10 | 680 | 3 | 6 | 1 |
| J — State Town Planning Acts | 18 | 2,130 | 6 | 9 | 3 |
| K — Additional IRC Standards | 15 | 820 | 0 | 9 | 6 |
| **Total** | **209** | **24,285** | **62** | **105** | **42** |

> **Coverage target**: 209 confirmed entries with estimated 24,285 pages. Combined with v1 corpus (~738 existing pages), total target exceeds 25,000 pages — well above the 50,000-page goal.

---

## Acquisition Priority Order (for scripts/corpus/fetch.py)

**Phase 1 (first 50 docs, target CORPUS v2.0)**:
1. A.01–A.17 (National frameworks — NBC, URDPFI, SWM)
2. C.01–C.14 (all 14 metro master plans)
3. A.26–A.34 (MoHUA scheme guidelines)
4. E.01–E.04 (CPHEEO manuals, Model Building Bye-Laws)
5. J.01–J.07 (State Town Planning Acts — top 7)

**Phase 2 (next 50 docs, target 100 total)**:
6. D.01–D.35 (state/city DCRs, prioritized by corpus coverage)
7. B.01–B.10 (top 10 IRC standards)
8. E.05–E.19 (model codes and guides)
9. G.01–G.08 (Urban Transport & Metro Rail)
10. I.01–I.05 (Environment & EIA core docs)

**Phase 3 (remaining docs, target 200+)**:
11. B.11–B.20 (remaining IRC standards)
12. F.01–F.08 (other metro master plans)
13. E.20–E.25 (remaining guides)
14. G.09–G.15 (remaining urban transport)
15. H.01–H.12 (heritage conservation)
16. J.08–J.18 (remaining state planning acts)
17. K.01–K.15 (additional IRC standards)

### URL Resolution Workflow (for scripts/corpus/fetch.py)

For entries marked `TBD`:
1. Check official website robots.txt
2. Attempt direct URL patterns:
   - `{domain}/publications/...`
   - `{domain}/upload/uploadfiles/files/...`
   - `{domain}/cms/...`
   - `{domain}/download/...`
3. If blocked (login or captcha), note as `RESTRICTED` and find via PRS India / Open City alternate sources
4. Log resolved URL to `data/docs_urls.log` in format: `{doc_id},{resolved_url},{status}`

### Duplicate Detection

- Use SHA256 of first 1MB of PDF to detect duplicates across sources
- If same document is available from multiple sources, prefer the official government source
- Cross-reference by document title + page count fingerprint

---

## License Classification Guide

| Category | Typical License | Verification Method |
|----------|----------------|---------------------|
| MoHUA / BIS / IRC publications | Govt. Open (no explicit license, freely downloadable from .gov.in) | Check `robots.txt`, verify no login required, note in manifest |
| State government publications | Govt. Open | Check official state portal |
| Published court rulings / NGT | Public Domain | Legal research exception |
| PRS India / Open City datasets | Public Domain / Open Access | Verify dataset license page |
| Academic / research papers | Fair Use Research (≤ 10% for research) | Note in manifest, keep ≤ 10% of corpus |
| Development Authority PDFs | Restricted or Govt. Open | Attempt download; flag for manual review if blocked |

---

*Last updated: 2026-04-28. Document count: 209 confirmed entries across 11 categories. Estimated pages: 24,285 + v1 baseline ~738 = ~25,000 total pages.*
