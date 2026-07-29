# Partial failure pilot — round 3

Каждый кандидат получил три внешних анонимных review. Автор не проверял свой
кандидат.

| Candidate | External scores | Maximin | Confirmed veto | Verdict |
|---|---|---:|---|---|
| D1 Uncertainty Cut | 0.58, 0.82, 0.43 | 0.43 | C1/C2 | INVALID |
| D2 Acceptance Closure | 0.40, 0.55, 0.91 | 0.40 | C1/C4, C6 | INVALID |
| D3 Effect Capability | 0.64, 0.74, 0.38 | 0.38 | C1, C3, C4 | INVALID |
| D4 Uncertainty Escrow | 0.48, 0.89, 0.91 | 0.48 | C1/C4 | INVALID |

## Подтверждённые контрпримеры

- **D1:** исполненный PRE-заказ с потерянным ACK повторяется без старого ID;
  сервер выдаёт новый POST-ID и разрешает второй эффект.
- **D2:** сертификат доказывает accepted-set, но не атомарность
  `effect + outcome receipt`; повтор с тем же ACC может дать второй эффект.
  Если pre-ACK history отсутствует, ACC нельзя выпустить в пределах C6.
- **D3:** crash между `CLAIMED`, capability delivery и effect оставляет
  stranded claim либо требует небезопасной переэмиссии; version isolation не
  входит в механизм.
- **D4:** durable pending ACK теряется; keyless retry создаёт вторую очередь.
  Последующий release двух записей даёт двойной эффект.

## Разрешённый ремонт D4

Ремонт заменил release keyless entries на вечный `AMBIGUOUS_INTENT`. Два мира
остаются неразличимы:

1. один заказ + lost ACK + retry;
2. два намеренно одинаковых заказа.

Выпустить две записи нарушает C1 в первом мире; слить их — C1 во втором;
выпустить ноль сохраняет safety, но не доказывает C6. Автор ремонта вернул
`REJECT`, не добавляя новый оператор.
