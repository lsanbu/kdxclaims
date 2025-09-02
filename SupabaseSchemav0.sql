-- =========================================================
-- Extensions (UUID + case-insensitive emails)
-- =========================================================
create extension if not exists pgcrypto;
create extension if not exists citext;

-- =========================================================
-- Root user profile (one per human)
-- =========================================================
create table if not exists app_users (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  full_name text,
  employer  text
);

-- Telegram identity (1:1)
create table if not exists user_telegram (
  user_id uuid primary key references app_users(id) on delete cascade,
  telegram_user_id bigint unique not null,
  telegram_chat_id bigint not null,
  created_at timestamptz not null default now()
);

-- Email identity (0..1)
create table if not exists user_email (
  user_id uuid primary key references app_users(id) on delete cascade,
  email citext unique not null,
  email_verified boolean not null default false,
  created_at timestamptz not null default now()
);

-- Phone identity (0..1) - store in E.164 format (+91...)
create table if not exists user_phone (
  user_id uuid primary key references app_users(id) on delete cascade,
  phone text unique not null,
  phone_verified boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists idx_user_telegram_tgid on user_telegram(telegram_user_id);
create index if not exists idx_user_email_email on user_email(email);
create index if not exists idx_user_phone_phone on user_phone(phone);

-- =========================================================
-- Claims (Fuel + Driver Salary) — lean, future-proof
-- =========================================================
create table if not exists claims (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references app_users(id) on delete cascade,

  -- 'fuel' | 'driver_salary'
  claim_type text not null check (claim_type in ('fuel','driver_salary')),

  -- common
  claim_date date not null,                 -- fuel: txn date; salary: voucher/payment date
  claim_time time,                          -- fuel optional
  total_rs numeric(10,2) not null,
  vendor text,                              -- fuel: station (also see station field); salary: can be 'Driver Salary'
  reference_no text,                        -- fuel: Inv/Receipt No; salary: voucher no or internal ref
  notes text,
  image_path text,                          -- storage key
  image_url text,                           -- public/signed URL (optional)
  status text not null default 'parsed',    -- parsed | needs_review | approved | exported
  created_at timestamptz not null default now(),

  -- fuel-specific (nullable for salary)
  station text,
  rate_rs_per_l numeric(8,2),
  volume_l numeric(9,3),
  odometer_km integer,
  calc_amount_rs numeric(10,2),
  variance_rs numeric(10,2),

  -- driver-salary-specific (nullable for fuel)
  period_start date,
  period_end date,
  days_worked integer,
  daily_rate_rs numeric(10,2),
  advances_rs numeric(10,2)
);

-- Helpful indexes for scale
create index if not exists idx_claims_user_date on claims(user_id, claim_date);
create index if not exists idx_claims_type_date on claims(claim_type, claim_date);
create index if not exists idx_claims_created_at on claims(created_at);

-- Optional sanity checks (won't block inserts; uncomment if desired)
-- alter table claims add constraint chk_fuel_amount
--   check (claim_type <> 'fuel' or total_rs >= 0);
-- alter table claims add constraint chk_salary_period
--   check (claim_type <> 'driver_salary' or (period_start is null) or (period_end is null) or (period_end >= period_start));

-- =========================================================
-- Row Level Security (RLS)
-- (For web: use Supabase Auth; for Telegram service_role, RLS is bypassed)
-- =========================================================
alter table app_users enable row level security;
alter table claims    enable row level security;

-- app_users: a user can read/update their own profile (when authenticated)
drop policy if exists app_users_owner_select on app_users;
create policy app_users_owner_select
on app_users for select
using (id = auth.uid());

drop policy if exists app_users_owner_update on app_users;
create policy app_users_owner_update
on app_users for update
using (id = auth.uid())
with check (id = auth.uid());

-- claims: owner-only access
drop policy if exists claims_owner_all on claims;
create policy claims_owner_all
on claims
for all
using (user_id = auth.uid())
with check (user_id = auth.uid());

-- =========================================================
-- (Optional) Helper view for month exports (merge both types)
-- =========================================================
create or replace view v_claims_export as
select
  c.id,
  c.user_id,
  c.claim_type,
  c.claim_date,
  c.claim_time,
  coalesce(c.station, c.vendor)        as place_or_payee,
  c.reference_no,
  c.rate_rs_per_l,
  c.volume_l,
  c.total_rs,
  c.odometer_km,
  c.period_start,
  c.period_end,
  c.days_worked,
  c.daily_rate_rs,
  c.advances_rs,
  c.notes,
  c.image_url,
  c.status,
  c.created_at
from claims c;

-- =========================================================
-- (Optional) Seed a system user for quick local testing
--  - REMOVE in production
-- =========================================================
-- with ins as (
--   insert into app_users (id, full_name, employer)
--   values ('00000000-0000-0000-0000-000000000001'::uuid, 'Demo User', 'KDx')
--   on conflict do nothing
--   returning id
-- )
-- insert into claims (user_id, claim_type, claim_date, claim_time, total_rs, station, reference_no, rate_rs_per_l, volume_l, calc_amount_rs, variance_rs, notes)
-- select '00000000-0000-0000-0000-000000000001'::uuid, 'fuel', '2025-08-24', '17:18', 309.76, 'Bharat Petroleum', '243622025H101809', 100.90, 3.07, 309.76, 0.00, 'Seed row'
-- where not exists (select 1 from claims limit 1);

-- Enable RLS
alter table app_users enable row level security;
alter table claims enable row level security;
alter table user_telegram enable row level security;
alter table user_email enable row level security;
alter table user_phone enable row level security;

-- Claims: owner-only access (when using Supabase Auth web flow)
create policy claims_owner_all
on claims
for all
using (user_id = auth.uid())
with check (user_id = auth.uid());

-- app_users: self-access only
create policy app_users_owner_select
on app_users for select
using (id = auth.uid());

create policy app_users_owner_update
on app_users for update
using (id = auth.uid())
with check (id = auth.uid());

-- Enable RLS
alter table app_users enable row level security;
alter table claims enable row level security;
alter table user_telegram enable row level security;
alter table user_email enable row level security;
alter table user_phone enable row level security;

-- Claims: owner-only access (when using Supabase Auth web flow)
create policy claims_owner_all
on claims
for all
using (user_id = auth.uid())
with check (user_id = auth.uid());

-- app_users: self-access only
create policy app_users_owner_select
on app_users for select
using (id = auth.uid());

create policy app_users_owner_update
on app_users for update
using (id = auth.uid())
with check (id = auth.uid());

-- =========================================================
-- Seed demo user (replace values with your own test Telegram ID/chat_id)
-- =========================================================

-- 1. Insert demo user
insert into app_users (id, full_name, employer)
values ('00000000-0000-0000-0000-000000000001'::uuid, 'Demo User', 'KDx Labs')
on conflict (id) do nothing;

-- 2. Link Telegram identity
-- Replace 123456789 with your actual telegram_user_id
-- Replace 987654321 with your actual telegram_chat_id
insert into user_telegram (user_id, telegram_user_id, telegram_chat_id)
values (
  '00000000-0000-0000-0000-000000000001'::uuid,
  7055992162,
  7055992162
)
on conflict (user_id) do nothing;

-- 3. Optionally link email / phone (not required for test)
insert into user_email (user_id, email, email_verified)
values (
  '00000000-0000-0000-0000-000000000001'::uuid,
  'demo@kdxclaims.in',
  true
)
on conflict (user_id) do nothing;

insert into user_phone (user_id, phone, phone_verified)
values (
  '00000000-0000-0000-0000-000000000001'::uuid,
  '+919876543210',
  true
)
on conflict (user_id) do nothing;

-- 4. Insert a sample fuel claim
insert into claims (
  user_id, claim_type, claim_date, claim_time,
  total_rs, station, reference_no, rate_rs_per_l, volume_l, calc_amount_rs, variance_rs, notes
)
values (
  '00000000-0000-0000-0000-000000000001'::uuid,
  'fuel',
  '2025-08-24',
  '17:18',
  309.76,
  'Bharat Petroleum',
  '243622025H101809',
  100.90,
  3.07,
  309.76,
  0.00,
  'Seed test row'
);

delete from user_telegram;

ALTER TABLE claims
  ADD CONSTRAINT claim_type_check
  CHECK (claim_type IN ('fuel', 'driver_salary', 'insurance', 'service', 'accessories'));

BEGIN;

-- 1) Drop any existing CHECK constraint on claims.claim_type (name-agnostic)
DO $$
DECLARE
  c record;
BEGIN
  FOR c IN
    SELECT conname
    FROM pg_constraint
    WHERE conrelid = 'public.claims'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%claim_type%'
  LOOP
    EXECUTE format('ALTER TABLE public.claims DROP CONSTRAINT %I;', c.conname);
  END LOOP;
END $$;

-- 2) Add the new allowed set for car-maintenance claims
ALTER TABLE public.claims
  ADD CONSTRAINT claims_claim_type_check
  CHECK (
    claim_type IN ('fuel','driver_salary','insurance','service','accessories')
  );

COMMIT;

-- What constraints are on claims now?
SELECT conname, pg_get_constraintdef(oid) AS def
FROM pg_constraint
WHERE conrelid = 'public.claims'::regclass
ORDER BY conname;

-- What claim types exist in your data?
SELECT claim_type, count(*) FROM public.claims GROUP BY 1 ORDER BY 1;

create table if not exists user_prefs (
  user_id uuid primary key references app_users(id) on delete cascade,
  cutoff_day int not null default 15,         -- corporate cycle “closes” on this day
  tz text not null default 'Asia/Kolkata',
  weekly_reminder boolean not null default true,
  daily_pre_cutoff boolean not null default true,  -- 5 days before cutoff → daily
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- convenience: set a default row when a Telegram link is created (optional)
create or replace function fn_init_user_prefs()
returns trigger as $$
begin
  insert into user_prefs(user_id) values (new.user_id)
  on conflict (user_id) do nothing;
  return new;
end; $$ language plpgsql;

drop trigger if exists trg_init_user_prefs on user_telegram;
create trigger trg_init_user_prefs after insert on user_telegram
for each row execute function fn_init_user_prefs();

create index if not exists idx_claims_user_date on claims (user_id, claim_date);
create index if not exists idx_ut_telegram_user on user_telegram (telegram_user_id);

