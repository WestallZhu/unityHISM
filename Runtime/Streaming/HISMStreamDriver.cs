
// HISMStreamDriver.cs
#nullable enable
using Unity.Mathematics;
using UnityEngine;

namespace Renderloom.HISM.Streaming
{
    public sealed class HISMStreamDriver : MonoBehaviour
    {
        [Header("Paths")]
        public string ManifestPath;
        public string RootPath;
        public ReadMode Mode = ReadMode.PackFile;

        [Header("Streaming Params")]
        public float CellSize = 64f;
        public float LoadDist = 256f;
        public float UnloadBias = 64f;
        public int   MaxNewLoadsPerFrame = 4;
        public int   MaxIOBytesPerFrame = 16 * 1024 * 1024;

        [Header("Backend")]
        public bool UseNullBackend = true;
        IHISMBackend? _backend;
        HISMStreamManager? _mgr;

        void Awake()
        {
            _backend = UseNullBackend ? new NullHISMBackend() : new HISMSystem(); // replace with your backend
            _mgr = new HISMStreamManager(_backend)
            {
                CellSize = CellSize,
                LoadDist = LoadDist,
                UnloadBias = UnloadBias,
                MaxNewLoadsPerFrame = MaxNewLoadsPerFrame,
                MaxIOBytesPerFrame = MaxIOBytesPerFrame,
                Mode = Mode
            };
            _mgr.LoadManifest(ManifestPath, RootPath, Mode);
            Debug.Log($"[HISM] SR Manifest loaded. Cells={_mgr.ManifestHeader.CellCount}, CellSize={_mgr.ManifestHeader.CellSize}");
        }

        void Update()
        {
            if (_mgr == null) return;
            var cam = Camera.main != null ? (float3)Camera.main.transform.position : float3.zero;
            _mgr.UpdateStreaming(cam);
        }

        void OnDestroy()
        {
            _mgr?.Dispose();
            _mgr = null;
        }

#if UNITY_EDITOR
        void OnDrawGizmosSelected()
        {
            if (_mgr == null) return;
            var cam = Camera.main != null ? (float3)Camera.main.transform.position : float3.zero;
            float r1 = LoadDist;
            float r2 = LoadDist + UnloadBias;
            DrawDiscXZ((Vector3)cam, r1, new Color(0,1,0,0.1f));
            DrawDiscXZ((Vector3)cam, r2, new Color(1,0.5f,0,0.08f));
        }

        static void DrawDiscXZ(Vector3 c, float r, Color col)
        {
            Gizmos.color = col;
            const int seg = 96;
            Vector3 p0 = c + new Vector3(r, 0, 0);
            for (int i = 1; i <= seg; i++)
            {
                float ang = (i / (float)seg) * Mathf.PI * 2f;
                Vector3 p1 = c + new Vector3(Mathf.Cos(ang) * r, 0, Mathf.Sin(ang) * r);
                Gizmos.DrawLine(p0, p1);
                p0 = p1;
            }
        }
#endif
    }
}
