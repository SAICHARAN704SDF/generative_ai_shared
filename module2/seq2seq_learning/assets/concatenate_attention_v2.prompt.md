# Diagram generation record

Generated on 2026-09-20 with the built-in image generation tool.

## Final prompt

Use case: scientific-educational
Create a polished wide landscape educational network diagram on white background titled "Concatenate attention in an LSTM encoder–decoder". Large crisp readable mathematical labels, generous spacing, clear arrowheads. Three panels: blue encoder, purple attention, green decoder, orange context paths. No decorative graphics.

LEFT: "Encoder — run once". Four source tokens das, buch, ist, rot each feed their own LSTM encoder cell. Four LSTM cells connected left to right by recurrent arrows. Each cell outputs h₁, h₂, h₃, h₄ upward or downward into an encoder memory H=(h₁,…,h₄). Memory has TWO outgoing arrows: one to attention scoring, one to weighted sum. Caption "Reuse H at every decoder step". Small note "Initial decoder states come from final encoder states (bridge if needed)."

MIDDLE: "Attention at step t". Query s_{t−1} enters a box "Concatenate each pair [s_{t−1}; h_i]"; encoder memory enters that box too. Downward sequential boxes:
"FC + tanh" with u_{t,i}=tanh(W_a[s_{t−1};h_i]+b_a)
"Scalar alignment score" with e_{t,i}=v_a^T u_{t,i}
"Softmax over source positions" with α_{t,i}=exp(e_{t,i})/Σ_k exp(e_{t,k})
"Weighted sum" with r_t=Σ_i α_{t,i}h_i.
The memory H arrow also enters weighted sum directly. Context r_t exits weighted sum as an orange line branching into TWO inputs in decoder panel.

RIGHT: "Decoder at step t". Previous target embedding E_y(y_{t−1}) feeds input concatenation [E_y(y_{t−1});r_t]. Orange context enters this concatenation. Concatenation feeds Decoder LSTM. Previous recurrent states (s_{t−1},m_{t−1}) feed LSTM separately. LSTM outputs updated hidden and cell states (s_t,m_t), with a recurrent arrow labeled "to step t+1". Only s_t branches into a second concatenation [s_t;r_t], alongside the SECOND orange context branch. This second concatenation feeds vocabulary projection ℓ_t=W_o[s_t;r_t]+b_o, then "Vocabulary softmax" p_t=softmax(ℓ_t), then "Select or sample token y_t". No cell-state arrow into vocabulary projection.

Bottom readable explanatory strip: "At t+1: query = s_t; recurrent states = (s_t,m_t); recompute attention using the same encoder outputs." Second line "r_t: attention context • s_t: decoder hidden state • m_t: decoder cell state".
Accuracy: attention is computed BEFORE decoder update using previous hidden state. Context affects BOTH LSTM state update and vocabulary projection. Never label softmax output as token directly. All arrows must point into destination boxes; avoid crossed lines and overlapping text. Draw all encoder cells explicitly and label as LSTM.
