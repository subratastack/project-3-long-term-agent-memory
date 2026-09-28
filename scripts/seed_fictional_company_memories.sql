-- Seed raw memory events for two fictional companies.
--
-- Prerequisite:
--   uv run alembic upgrade head
--
-- Run from the repository root:
--   docker compose exec -T postgres psql -U agent_memory -d agent_memory \
--     < scripts/seed_fictional_company_memories.sql
--
-- This script deliberately stops at memory_events. Use the API afterward:
--   POST /tenants/{tenant_id}/events/{event_id}/ingest
--   POST /tenants/{tenant_id}/memories/index
--
-- Fixed UUIDs and ON CONFLICT make the seed safe to run repeatedly.

BEGIN;

-- Register the companies so their events can be processed by the
-- tenant-scoped API. These are the only non-event rows inserted here.
INSERT INTO tenants (tenant_id, name, description)
VALUES
    (
        '11111111-1111-4111-8111-111111111111',
        'CartNest Customer Care',
        'Fictional ecommerce customer-service company supporting online shoppers and merchants.'
    ),
    (
        '22222222-2222-4222-8222-222222222222',
        'ClassForge Engineering',
        'Fictional engineering team building a multi-tenant school management system.'
    )
ON CONFLICT (tenant_id) DO NOTHING;

INSERT INTO memory_events (
    event_id, tenant_id, source_type, source_reference, content,
    observed_at, actor_id, metadata
)
VALUES
    -- CartNest Customer Care: ecommerce support events.
    (
        '10000000-0000-4000-8000-000000000001',
        '11111111-1111-4111-8111-111111111111',
        'configuration', 'seed://cartnest/policies/returns-v3',
        'Returns policy v3 permits eligible unused products to be returned within 30 calendar days of delivery. Final-sale and personalized items are excluded.',
        '2026-09-01 09:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"CartNest Customer Care","topic":"returns"}'
    ),
    (
        '10000000-0000-4000-8000-000000000002',
        '11111111-1111-4111-8111-111111111111',
        'configuration', 'seed://cartnest/runbooks/identity-check',
        'Before disclosing or changing an order, support must match the order number plus either the account email address or the delivery postal code.',
        '2026-09-02 10:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"CartNest Customer Care","topic":"identity_verification"}'
    ),
    (
        '10000000-0000-4000-8000-000000000003',
        '11111111-1111-4111-8111-111111111111',
        'configuration', 'seed://cartnest/security/payment-data',
        'Agents must never ask for or store a full payment-card number or CVV. Investigations use the payment-provider reference and last four digits only.',
        '2026-09-03 11:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"CartNest Customer Care","topic":"payment_security"}'
    ),
    (
        '10000000-0000-4000-8000-000000000004',
        '11111111-1111-4111-8111-111111111111',
        'configuration', 'seed://cartnest/service-levels/2026',
        'Acknowledge cancellation or delivery-address change requests within 15 minutes. Acknowledge all other customer tickets within four business hours.',
        '2026-09-04 09:30:00+00', NULL,
        '{"seed":"fictional_companies","company":"CartNest Customer Care","topic":"service_levels"}'
    ),
    (
        '10000000-0000-4000-8000-000000000005',
        '11111111-1111-4111-8111-111111111111',
        'configuration', 'seed://cartnest/runbooks/damaged-item',
        'For a damaged item, collect the order ID and photos of the item and packaging, record whether it is unsafe, then offer the eligible replacement or refund.',
        '2026-09-05 08:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"CartNest Customer Care","topic":"damaged_item"}'
    ),
    (
        '10000000-0000-4000-8000-000000000006',
        '11111111-1111-4111-8111-111111111111',
        'system_event', 'seed://cartnest/incidents/carrier-delay-2026-09',
        'Carrier incident CS-1842 is causing two-to-three-day delivery delays for western-region economy shipments through 2026-10-05.',
        '2026-09-24 14:15:00+00', NULL,
        '{"seed":"fictional_companies","company":"CartNest Customer Care","topic":"shipping_delay","incident_id":"CS-1842"}'
    ),
    (
        '10000000-0000-4000-8000-000000000007',
        '11111111-1111-4111-8111-111111111111',
        'configuration', 'seed://cartnest/support/hours',
        'Chat and email support operate daily from 08:00 to 22:00 UTC. Phone support operates Monday through Friday from 09:00 to 18:00 UTC.',
        '2026-09-06 12:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"CartNest Customer Care","topic":"support_hours"}'
    ),
    (
        '10000000-0000-4000-8000-000000000008',
        '11111111-1111-4111-8111-111111111111',
        'configuration', 'seed://cartnest/runbooks/escalation',
        'Immediately route safety hazards, legal threats, suspected account takeover, and chargeback notices to a senior specialist without promising an outcome.',
        '2026-09-07 13:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"CartNest Customer Care","topic":"escalation"}'
    ),

    -- ClassForge Engineering: school-management development events.
    (
        '20000000-0000-4000-8000-000000000001',
        '22222222-2222-4222-8222-222222222222',
        'configuration', 'seed://classforge/architecture/platform',
        'The API is FastAPI, the web client is React with TypeScript, and PostgreSQL is the system of record. Redis is used only for cache and queues.',
        '2026-09-01 09:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"ClassForge Engineering","topic":"architecture"}'
    ),
    (
        '20000000-0000-4000-8000-000000000002',
        '22222222-2222-4222-8222-222222222222',
        'configuration', 'seed://classforge/engineering/pull-requests',
        'Branch from main, link the work item, add tests, obtain one approval, pass CI, and squash merge. Database changes also require a rollback note.',
        '2026-09-02 10:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"ClassForge Engineering","topic":"pull_requests"}'
    ),
    (
        '20000000-0000-4000-8000-000000000003',
        '22222222-2222-4222-8222-222222222222',
        'configuration', 'seed://classforge/security/student-data',
        'Never put student names, contact details, grades, attendance notes, or guardian data in application logs. Use opaque IDs and redact exported diagnostics.',
        '2026-09-03 11:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"ClassForge Engineering","topic":"student_privacy"}'
    ),
    (
        '20000000-0000-4000-8000-000000000004',
        '22222222-2222-4222-8222-222222222222',
        'configuration', 'seed://classforge/architecture/tenant-isolation',
        'Every school-owned table includes school_id, repositories require school scope, and integration tests must prove records cannot cross school boundaries.',
        '2026-09-04 09:30:00+00', NULL,
        '{"seed":"fictional_companies","company":"ClassForge Engineering","topic":"tenant_isolation"}'
    ),
    (
        '20000000-0000-4000-8000-000000000005',
        '22222222-2222-4222-8222-222222222222',
        'configuration', 'seed://classforge/domain/ownership',
        'Enrollment owns students and guardians; Academics owns courses, timetables, and grades; Operations owns attendance, transport, and fee collection.',
        '2026-09-05 08:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"ClassForge Engineering","topic":"domain_ownership"}'
    ),
    (
        '20000000-0000-4000-8000-000000000006',
        '22222222-2222-4222-8222-222222222222',
        'system_event', 'seed://classforge/sprints/2026-09',
        'Sprint 2026-09 prioritizes guardian notification reliability: eliminate duplicate absence alerts and expose delivery status before the October pilot.',
        '2026-09-24 14:15:00+00', NULL,
        '{"seed":"fictional_companies","company":"ClassForge Engineering","topic":"current_sprint","sprint":"2026-09"}'
    ),
    (
        '20000000-0000-4000-8000-000000000007',
        '22222222-2222-4222-8222-222222222222',
        'configuration', 'seed://classforge/operations/jobs',
        'Attendance summaries run at 18:00 in each school timezone. Fee reminders run at 08:00 local time and skip holidays configured on the school calendar.',
        '2026-09-06 12:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"ClassForge Engineering","topic":"scheduled_jobs"}'
    ),
    (
        '20000000-0000-4000-8000-000000000008',
        '22222222-2222-4222-8222-222222222222',
        'configuration', 'seed://classforge/runbooks/incidents',
        'Page the on-call engineer for login failure, cross-school exposure, notification backlog, or grade-write errors. Freeze deployments and preserve audit evidence for severity-one incidents.',
        '2026-09-07 13:00:00+00', NULL,
        '{"seed":"fictional_companies","company":"ClassForge Engineering","topic":"incident_response"}'
    )
ON CONFLICT (tenant_id, event_id) DO NOTHING;

COMMIT;

-- Verification:
-- SELECT metadata->>'company' AS company, count(*) AS events
-- FROM memory_events
-- WHERE metadata->>'seed' = 'fictional_companies'
-- GROUP BY metadata->>'company'
-- ORDER BY company;
