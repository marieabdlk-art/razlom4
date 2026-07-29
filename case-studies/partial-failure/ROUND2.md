# Partial failure pilot — round 2

Frozen proposals были анонимизированы. Каждая роль атаковала три чужих
механизма, сохранила чужой инвариант и отказалась от одной собственной
assumption.

## D1 — Uncertainty Cut (`amputate`)

Отброшена assumption о том, что каждый старый заказ обнаружим. Вместо деления
узлов вводится временная граница identity: существовавшие или повторяемые ключи
становятся `PRE/UNKNOWN`; исполняются только сервером выданные `POST(epoch)`.

Новый оператор: неизвестность становится свойством исторического домена, а не
всего кластера. Ablation: без epoch-scoped ID новый заказ неотличим от retry.

## D2 — Acceptance-Closure Certificate (`reify`)

Утверждение о полноте превращается в объект
`ACC(epoch, log_prefix_hash, accepted_id_set_commitment, migration_version)`.
Только сертификат, доказывающий pre-ACK durable append и единственную canonical
identity каждого заказа, даёт execution authority. Без ACC — BLOCK.

Ablation: наличие журнала снова ошибочно принимается за доказательство его
полноты.

## D3 — Effect Capability (`reify`)

Право на бизнес-эффект переносится к downstream boundary. Атомарный
`claim(order_id)` возвращает прежний `APPLIED` receipt либо единственную
capability; только capability разрешает эффект.

Ablation: потерянный ACK снова позволяет повторный эффект.

## D4 — Uncertainty Escrow (`delay_or_expire`)

Отделить приём от исполнения. За две минуты endpoint durably сохраняет payload
и возвращает `ACCEPTED_PENDING_PROOF`; потенциальные retries не исполняются до
появления provenance. Legacy-кластер не мутируется.

Ablation: без delay ambiguous retry может примениться дважды; expiry запрещён,
потому что теряет принятый заказ.

## Новый общий вопрос

Все четыре мутации обнаружили одну и ту же более глубокую границу: epoch,
журнал, очередь и capability не создают ретроактивно semantic identity заказа.
Нужно различить retry после потерянного ACK и второй намеренно одинаковый заказ.
