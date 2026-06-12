{{/*
SAINT-OPS Helm helpers
*/}}
{{- define "saint-ops.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "saint-ops.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "saint-ops.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end }}

{{- define "saint-ops.labels" -}}
helm.sh/chart: {{ include "saint-ops.chart" . }}
{{ include "saint-ops.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: saint-ops
{{- end }}

{{- define "saint-ops.selectorLabels" -}}
app.kubernetes.io/name: {{ include "saint-ops.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "saint-ops.backend.selectorLabels" -}}
{{ include "saint-ops.selectorLabels" . }}
app.kubernetes.io/component: backend
{{- end }}

{{- define "saint-ops.frontend.selectorLabels" -}}
{{ include "saint-ops.selectorLabels" . }}
app.kubernetes.io/component: frontend
{{- end }}

{{- define "saint-ops.mlflow.selectorLabels" -}}
{{ include "saint-ops.selectorLabels" . }}
app.kubernetes.io/component: mlflow
{{- end }}

{{- define "saint-ops.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "saint-ops.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "saint-ops.mlflow.trackingUri" -}}
{{- if .Values.backend.mlflow.trackingUri }}
{{- .Values.backend.mlflow.trackingUri }}
{{- else }}
{{- printf "http://%s-mlflow:%v" (include "saint-ops.fullname" .) (.Values.mlflow.service.port | int) }}
{{- end }}
{{- end }}

{{- define "saint-ops.backend.env" -}}
- name: DATA_DIR
  value: /app/data
- name: ARTIFACTS_DIR
  value: /app/artifacts
- name: MODELS_DIR
  value: /app/models
- name: STREAM_CSV
  value: /app/data/scenario_C_annulus.csv
- name: MLFLOW_TRACKING_URI
  value: {{ include "saint-ops.mlflow.trackingUri" . | quote }}
- name: MLFLOW_REGISTERED_MODEL
  value: {{ .Values.backend.mlflow.registeredModel | quote }}
- name: MLFLOW_MODEL_URI
  value: {{ .Values.backend.mlflow.modelUri | quote }}
- name: MLFLOW_SHADOW_MODEL_URI
  value: {{ .Values.backend.mlflow.shadowModelUri | quote }}
- name: MLFLOW_SHADOW_TRAFFIC_PCT
  value: {{ .Values.backend.mlflow.shadowTrafficPct | quote }}
- name: MLFLOW_EXPERIMENT_NAME
  value: {{ .Values.backend.mlflow.experimentName | quote }}
- name: MLFLOW_REQUIRE_APPROVAL
  value: {{ .Values.backend.mlflow.requireApproval | quote }}
- name: MLFLOW_PROMOTION_MIN_F1_DELTA
  value: {{ .Values.backend.mlflow.promotionMinF1Delta | quote }}
{{- range $k, $v := .Values.backend.env }}
- name: {{ $k }}
  value: {{ $v | quote }}
{{- end }}
{{- end }}

{{- define "saint-ops.storage.volumeName" -}}
{{- printf "%s-storage" (include "saint-ops.fullname" .) }}
{{- end }}
