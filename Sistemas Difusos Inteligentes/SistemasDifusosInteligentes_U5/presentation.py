from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

COLOR_PRIMARIO = RGBColor(15, 76, 129)    # Azul institucional
COLOR_TEXTO    = RGBColor(40, 40, 40)

def agregar_diapositiva_bullets(titulo, puntos):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    title_shape = slide.shapes.title
    title_shape.text = titulo
    title_shape.text_frame.paragraphs[0].font.color.rgb = COLOR_PRIMARIO
    title_shape.text_frame.paragraphs[0].font.bold = True
    title_shape.text_frame.paragraphs[0].font.size = Pt(32)

    tf = slide.placeholders[1].text_frame
    tf.clear()
    for i, punto in enumerate(puntos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = punto
        p.font.size = Pt(20)
        p.font.color.rgb = COLOR_TEXTO
        p.space_after = Pt(12)

# --- Diapositiva 1: Portada ---
slide1 = prs.slides.add_slide(prs.slide_layouts[0])
slide1.shapes.title.text = "AgroNash-FIS: Asignación Óptima de Agua Agrícola"
slide1.shapes.title.text_frame.paragraphs[0].font.color.rgb = COLOR_PRIMARIO
slide1.placeholders[1].text = (
    "Sistema Híbrido de Inferencia Difusa Mamdani y Negociación Difusa de Nash\n\n"
    "Maestría en Inteligencia Artificial | Instituto Tecnológico El Llano Aguascalientes"
)

# --- Diapositivas 2 a 8 ---
agregar_diapositiva_bullets("1. Planteamiento del Problema", [
    "Sobreexplotación de acuíferos y alta variabilidad climática en zonas agrícolas.",
    "Los sistemas clásicos (umbrales rígidos 0/1) fallan al modelar variables graduales.",
    "Conflicto frecuente: demanda hídrica superior al volumen disponible.",
    "Reto: equilibrar eficiencia técnica de riego con equidad en el reparto."
])

agregar_diapositiva_bullets("2. Objetivo e Innovación: Arquitectura en 2 Capas", [
    "Capa 1 (Evaluación Técnica - Mamdani): Determina la cuota ideal de cada parcela.",
    "Capa 2 (Resolución de Conflictos - Fuzzy Nash): Reparte agua bajo escasez.",
    "Diferenciador clave: une Lógica Difusa con Teoría de Juegos Cooperativa.",
    "Implementación interactiva dual en Python (Streamlit) y MATLAB (App Designer)."
])

agregar_diapositiva_bullets("3. Marco Teórico", [
    "Conjuntos Difusos (Zadeh, 1965): grados de pertenencia continuos en [0, 1].",
    "Modelo Mamdani: operadores min (AND), max (OR) y defusificación por centroide.",
    "Negociación Difusa de Nash: sustituye utilidades clásicas por satisfacción μ(x).",
    "Contraste de t-normas: Producto (Fuzzy Nash) vs. Mínimo (Bellman-Zadeh)."
])

agregar_diapositiva_bullets("4. Diseño del FIS: Variables y Funciones de Membresía", [
    "Entrada 1 - Estrés Hídrico (E) [0, 100%]: Bajo (Trap), Moderado (Tri), Severo (Trap).",
    "Entrada 2 - Disponibilidad Acuífero (D) [0, 100%]: Crítica (Trap), Media (Tri), Alta (Trap).",
    "Entrada 3 - Eficiencia de Riego (F) [0, 100%]: Baja (Rodado), Media (Aspersión), Alta (Goteo).",
    "Salida - Cuota de Asignación (C) [0, 100%]: Mínima, Moderada, Óptima (Triangulares)."
])

agregar_diapositiva_bullets("5. Base de Conocimiento (27 Reglas Exhaustivas 3×3×3)", [
    "Cobertura total sin zonas muertas en el universo de discurso.",
    "Criterio 1: Si el estrés es Bajo, se asigna cuota Mínima (evita desperdicio).",
    "Criterio 2: Si el acuífero está en estado Crítico, se restringe la extracción.",
    "Criterio 3: Se premia la Eficiencia Alta (goteo) en escenarios de estrés severo."
])

agregar_diapositiva_bullets("6. Módulo de Innovación: Negociación Difusa de Nash", [
    "Escenario: Presupuesto B = 1,000 m³ entre Zona A (demanda 700) y Zona B (demanda 600).",
    "Puntos de desacuerdo (supervivencia): d1 = 200 m³ y d2 = 150 m³.",
    "Solución Fuzzy Nash (Producto μ1 · μ2): A = 525 m³, B = 475 m³.",
    "Ventaja: logra un balance suave y superior a recortes proporcionales arbitrarios."
])

agregar_diapositiva_bullets("7. Demostración Interactiva y Superficie de Control 3D", [
    "Interfaz web en Python (Streamlit + Plotly) y nativa en MATLAB (uifigure).",
    "Superficie 3D interactiva: Estrés × Acuífero → Cuota (con Eficiencia ajustable).",
    "Curvas comparativas en tiempo real: Fuzzy Nash vs. Bellman-Zadeh.",
    "[Insertar aquí captura de pantalla de la app en ejecución]"
])

# --- Diapositiva 9: Tabla de Validación ---
slide9 = prs.slides.add_slide(prs.slide_layouts[5])
slide9.shapes.title.text = "8. Validación de Resultados (Esperado vs. Obtenido)"
slide9.shapes.title.text_frame.paragraphs[0].font.color.rgb = COLOR_PRIMARIO

datos_tabla = [
    ["Escenario de Prueba", "Entradas (E, D, F)", "Esperado", "Obtenido"],
    ["1. Sequía extrema + Riego ineficiente", "(85%, 15%, 30%)", "Mínima (<30%)", "14.8 %"],
    ["2. Estrés severo + Riego por goteo", "(85%, 55%, 90%)", "Óptima (>75%)", "82.4 %"],
    ["3. Cultivo sano + Acuífero lleno", "(15%, 85%, 70%)", "Mínima (<30%)", "14.8 %"],
    ["4. Condición promedio de temporada", "(50%, 50%, 60%)", "Moderada (~50%)", "50.0 %"],
]
tabla_shape = slide9.shapes.add_table(5, 4, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.2))
tabla = tabla_shape.table
for r_idx, fila in enumerate(datos_tabla):
    for c_idx, val in enumerate(fila):
        celda = tabla.cell(r_idx, c_idx)
        celda.text = val
        p = celda.text_frame.paragraphs[0]
        p.font.size = Pt(16)
        if r_idx == 0:
            p.font.bold = True

# --- Diapositiva 10: Conclusiones ---
agregar_diapositiva_bullets("9. Conclusiones y Trabajo Futuro", [
    "AgroNash-FIS traduce conocimiento experto agronómico en decisiones suaves y auditables.",
    "La Capa 2 (Fuzzy Nash) resuelve conflictos de escasez con rigor matemático.",
    "La validación dual (Python y MATLAB) confirma la reproducibilidad del modelo.",
    "Trabajo futuro: integración con sensores IoT y aprendizaje adaptativo con ANFIS."
])

prs.save("Presentacion_AgroNash_FIS.pptx")
print("Presentación generada exitosamente: Presentacion_AgroNash_FIS.pptx")