# PBIX Build Handoff

The repository now contains all source artifacts needed to assemble the research dashboard in Power BI Desktop:

- `dashboard_storyboard.md`
- `dashboard_data_dictionary.md`
- `dashboard_screenshot_pack/`
- all exported CSV tables from notebooks `05` and `06`

## What is ready

- page structure
- source tables
- recommended relationships
- KPI definitions
- page-level visual logic
- wireframe screenshots

## What still requires Power BI Desktop

- importing the CSVs into a `.pbix`
- building measures and visuals
- page layout polish
- slicer interactions
- final theme and formatting

## Recommended build order

1. Import all tables listed in `dashboard_storyboard.md`
2. Build the relationships from `dashboard_data_dictionary.md`
3. Create KPI measures first
4. Build page 4 and page 5 first because they hold the core research value
5. Finish with page 6 watchlist and page 1 overview

## Naming

When the Power BI Desktop build is created, save it as:

`powerbi/final_decision_dashboard.pbix`
